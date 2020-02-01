from pprint import pprint
import code
import time
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import csc_matrix
import heapq

log_info = lambda *a: print("Info:", *a)
log_warning = lambda *a: print("Warning:", *a)
log_error = lambda *a: print("ERROR:", *a)

def duration(ns):
    return ns / 1000000000

class Dummy:
    pass

class DummyWithSlots:
    def __init__(self, *args, **kw):
        for n, a in zip(self.__slots__, args):
            setattr(self, n, a)
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        name = type(self).__name__
        
        return name + repr({k: getattr(self, k) for k in self.__slots__})

class CellLoc(DummyWithSlots):
    __slots__ = ("x", "y", "legal_x", "legal_y", "rawx", "rawy", "locked", "global_")

class BoundingBox(DummyWithSlots):
    __slots__ = ("x0", "y0", "x1", "y1")

class ChainExtent(DummyWithSlots):
    __slots__ = ("x0", "y0", "x1", "y1")

class SpreaderRegion(DummyWithSlots):
    __slots__ = ("id", "x0", "y0", "x1", "y1", "cells", "bels")
    def overused(self):
        if self.bels < 4:
            return self.cells > self.bels
        else:
            beta = 0.9
            return self.cells > beta * self.bels

class HeAPPlacer:
    def __getattr__(self, name):
        return globals()[name]
    def __setattr__(self, name, value):
        globals()[name] = value

max_double = 999999999999999999999999.0

cfg = Dummy()
#cfg.alpha = ctx.setting<float>("placerHeap/alpha", 0.1)
#cfg.criticalityExponent = ctx.setting<int>("placerHeap/criticalityExponent", 2)
#cfg.timingWeight = ctx.setting<int>("placerHeap/timingWeight", 10)
#cfg.timing_driven = ctx.setting<bool>("timing_driven")
cfg.alpha = 0.1
cfg.criticalityExponent = 2
cfg.timingWeight = 10
cfg.timing_driven = False

availableBels = {}
net_crit = {}

max_x, max_y = 0, 0
fast_bels = []
bel_types = {}

nearest_row_with_bel = []
nearest_col_with_bel = []
constraint_region_bounds = {}

cell_locs = {}
place_cells = []
solve_cells = []
ioBufTypes = {"TRELLIS_IO"}


chain_root = {}
chain_size = {}
cell_offsets = {}

"""
    Context *ctx
    PlacerHeapCfg cfg
    int max_x = 0, max_y = 0
    std::vector<std::vector<std::vector<std::vector<BelId>>>> fast_bels
    std::unordered_map<IdString, std::tuple<int, int>> bel_types
    std::vector<std::vector<int>> nearest_row_with_bel
    std::vector<std::vector<int>> nearest_col_with_bel
    std::unordered_map<IdString, BoundingBox> constraint_region_bounds

    std::unordered_map<IdString, CellLocation> cell_locs
    # The set of cells that we will actually place. This excludes locked cells and children cells of macros/chains
    # (only the root of each macro is placed.)
    std::vector<CellInfo *> place_cells

    # The cells in the current equation being solved (a subset of place_cells in some cases, where we only place
    # cells of a certain type)
    std::vector<CellInfo *> solve_cells

    # For cells in a chain, this is the ultimate root cell of the chain (sometimes this is not constr_parent
    # where chains are within chains
    std::unordered_map<IdString, CellInfo *> chain_root
    std::unordered_map<IdString, int> chain_size

    # The offset from chain_root to a cell in the chain
    std::unordered_map<IdString, std::pair<int, int>> cell_offsets

    # Performance counting
    double solve_time = 0, cl_time = 0, sl_time = 0

    NetCriticalityMap net_crit
"""

solve_time = 0
cl_time = 0
sl_time = 0

class EquationSystem:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.Arow  = []
        self.Acol  = []
        self.Aval = []
        self.rhs = np.zeros((rows,))  # RHS vector

    def reset(self):
        self.__init__(self.rows, self.cols)

    def add_coeff(self, row, col, val):
        self.Arow.append(row)
        self.Acol.append(col)
        self.Aval.append(val)

    def add_rhs(self, row, val):
        self.rhs[row] += val

    def solve(self, x):
        A = csc_matrix((self.Aval, (self.Arow, self.Acol)),
            shape=(self.rows, self.cols)).toarray()
        x = cg(A, self.rhs, tol=1e-5, x0=x)
        return x

def place():
    global net_crit
    global solve_time
    global sl_time
    global cl_time

    startt = time.time_ns()

    place_constraints()
    build_fast_bels()
    seed_placement()
    update_all_chains()
    hpwl = total_hpwl()
    log_info("Creating initial analytic placement for {} cells, random placement wirelen = {}.".format(
             len(place_cells), int(hpwl)))

    for i in range(4):
        setup_solve_cells()
        build_solve_direction("x", -1)
        build_solve_direction("y", -1)
        update_all_chains()
        #
        hpwl = total_hpwl()
        log_info("    at initial placer iter {}, wirelen = {}".format(i, int(hpwl)))

    solved_hpwl = 0
    spread_hpwl = 0
    legal_hpwl = 0
    best_hpwl = 0x7fffffffffffffff
    iteration = 0
    stalled = 0

    solution = []         # std::vector<std::tuple<CellInfo *, BelId, PlaceStrength>>

    heap_runs = []        # std::vector<std::unordered_set<IdString>>
    all_celltypes = set() # std::unordered_set<IdString>
    ct_count = {}         # std::unordered_map<IdString, int>

    for cell in place_cells:
        if cell.type not in all_celltypes:
            heap_runs.append({cell.type})
            all_celltypes.add(cell.type)

        ct_count[cell.type] = ct_count.get(cell.type, 0)+1

    # If more than 98% of cells are one cell type, always solve all at once
    # Otherwise, follow full HeAP strategy of rotate&all
    for cc in ct_count.values():
        if cc >= 0.98 * len(place_cells):
            heap_runs = []
            break

    heap_runs.append(all_celltypes)

    # The main HeAP placer loop
    log_info("Running main analytical placer.")
    while (stalled < 5 and (solved_hpwl <= legal_hpwl * 0.8)):
        # Alternate between particular Bel types and all bels
        for run in heap_runs:
            run_startt = time.time_ns()

            setup_solve_cells(run)
            if len(solve_cells) == 0:
                continue
            # Heuristic: don't bother with threading below a certain size
            solve_startt = time.time_ns()

            #if (solve_cells.size() < 500):
            build_solve_direction("x", -1 if (iteration == 0) else iteration)
            build_solve_direction("y", -1 if (iteration == 0) else iteration)
            #else:
            #    boost::thread xaxis([&]() { build_solve_direction(False, (iteration == 0) ? -1 : iteration); })
            #    build_solve_direction(True, (iteration == 0) ? -1 : iteration)
            #    xaxis.join()

            solve_endt = time.time_ns()
            solve_time += duration(solve_endt - solve_startt)
            update_all_chains()
            solved_hpwl = total_hpwl()

            update_all_chains()
            for ty in sorted(run):
                CutSpreader(HeAPPlacer(), ty).run()

            update_all_chains()
            spread_hpwl = total_hpwl()
            legalise_placement_strict(True)
            update_all_chains()

            legal_hpwl = total_hpwl()
            run_stopt = time.time_ns()
            log_info("    at iteration #%d, type %s: wirelen solved = %d, spread = %d, legal = %d; time = %.02fs" % (
                     iteration + 1, ("ALL" if len(run)>1 else next(iter(run))), solved_hpwl,
                     spread_hpwl, legal_hpwl,
                     duration(run_stopt - run_startt)))

        if (cfg.timing_driven):
            net_crit = get_criticalities(ctx)

        if (legal_hpwl < best_hpwl):
            best_hpwl = legal_hpwl
            stalled = 0
            # Save solution
            del solution[:]
            for cell in ctx.cells:
                solution.append((cell.second, cell.second.bel, cell.second.belStrength))

        else:
            stalled += 1

        for cl in cell_locs.values():
            cl.legal_x = cl.x
            cl.legal_y = cl.y

        #ctx.yield()
        iteration += 1

    # Apply saved solution
    for cell, _bel, _strength in solution:
        if cell.bel:
            ctx.unbindBel(cell.bel)

    for cell, bel, strength in solution:
        assert bel
        ctx.bindBel(bel, cell, strength)

    for cell, cinfo in ctx.cells:
        if not cinfo.bel:
            log_error("Found unbound cell %s" % cell)
        if ctx.getBoundBelCell(cinfo.bel).name != cinfo.name:
            log_error("Found cell %s with mismatched binding" % cell)
            log_error(" {} != {}".format(ctx.getBoundBelCell(cinfo.bel).name, cinfo.name))
        #if (ctx.debug): # TODO
        #    log_info("AP soln: %s . %s\n", cell, ctx.getBelName(cinfo.bel))


    #ctx.unlock()
    endtt = time.time_ns()
    log_info("HeAP Placer Time: %.02fs" % duration(endtt - startt))
    log_info("  of which solving equations: %.02fs" % solve_time)
    log_info("  of which spreading cells: %.02fs" % cl_time)
    log_info("  of which strict legalisation: %.02fs" % sl_time)

    #ctx.check()

    #placer1_refine(ctx, Placer1Cfg(ctx))

    return True

def place_constraints():
    placed_cells = 0
    # Initial constraints placer
    for cell, cinfo in ctx.cells:
        cell = cinfo
        if "BEL" not in cell.attrs:
            continue

        loc_name = cell.attrs["BEL"]

        #bel = ctx.getBelByName(loc_name)
        #if bel:
        #    log_error("No Bel named \'%s\' located for "
        #              "this chip (processing BEL attribute on \'%s\')\n",
        #              loc_name.c_str(), cell.name)
        bel = loc_name

        bel_type = ctx.getBelType(bel)
        if (bel_type != cell.type):
            log_error("Bel \'%s\' of type \'%s\' does not match cell "
                      "\'%s\' of type \'%s\'\n",
                      loc_name.c_str(), bel_type, cell.name, cell.type)

        if not ctx.isValidBelForCell(cell, bel):
            log_error("Bel \'%s\' of type \'%s\' is not valid for cell "
                      "\'%s\' of type \'%s\'\n",
                      loc_name.c_str(), bel_type, cell.name, cell.type)

        bound_cell = ctx.getBoundBelCell(bel)
        if bound_cell:
            log_error("Cell \'%s\' cannot be bound to bel \'%s\' since it is already bound to cell \'%s\'\n",
                      cell.name, loc_name.c_str(), bound_cell.name)

        ctx.bindBel(bel, cell, STRENGTH_USER)
        placed_cells += 1

    log_info("Placed %d cells based on constraints." % placed_cells)
    #ctx.yield_()

def build_fast_bels():
    global max_x
    global max_y
    global nearest_row_with_bel
    global nearest_col_with_bel
    num_bel_types = 0
    for bel in ctx.getBels():
        ty = ctx.getBelType(bel)
        if ty in bel_types:
            bel_types[ty][1] += 1
        else:
            bel_types[ty] = [num_bel_types, 1]
            num_bel_types += 1

    for bel in ctx.getBels():
        if not ctx.checkBelAvail(bel):
            continue
        loc = ctx.getBelLocation(bel)
        ty = ctx.getBelType(bel)
        type_idx, _ = bel_types[ty]
        def stretch_array(a, i):
            while len(a) <= i:
                a.append([])
        stretch_array(fast_bels, type_idx)
        stretch_array(fast_bels[type_idx], loc.x)
        stretch_array(fast_bels[type_idx][loc.x], loc.y)
        max_x = max(max_x, loc.x)
        max_y = max(max_y, loc.y)
        fast_bels[type_idx][loc.x][loc.y].append(bel)

    nearest_row_with_bel = [[-1]*(max_y + 1) for ty in range(num_bel_types)]
    nearest_col_with_bel = [[-1]*(max_x + 1) for ty in range(num_bel_types)]

    for bel in ctx.getBels():
        if not ctx.checkBelAvail(bel):
            continue
        loc = ctx.getBelLocation(bel)
        type_idx, _ = bel_types[ctx.getBelType(bel)]
        nr = nearest_row_with_bel[type_idx]
        nc = nearest_col_with_bel[type_idx]

        # Traverse outwards through nearest_row_with_bel and nearest_col_with_bel, stopping once
        # another row/col is already recorded as being nearer
        for x in range(loc.x, max_x):
            if nc[x] != -1 and abs(loc.x - nc[x]) <= (x - loc.x):
                break
            nc[x] = loc.x

        for x in range(loc.x - 1, -1, -1):
            if nc[x] != -1 and abs(loc.x - nc[x]) <= (loc.x - x):
                break
            nc[x] = loc.x

        for y in range(loc.y, max_y):
            if nr[y] != -1 and abs(loc.y - nr[y]) <= (y - loc.y):
                break
            nr[y] = loc.y

        for y in range(loc.y - 1, -1, -1):
            if nr[y] != -1 and abs(loc.y - nr[y]) <= (loc.y - y):
                break
            nr[y] = loc.y

    # Determine bounding boxes of region constraints
    for region, r in ctx.region:
        bb = BoundingBox()
        if r.constr_bels:
            bb.x0 =  0x7fffffff
            bb.x1 = -0x80000000
            bb.y0 =  0x7fffffff
            bb.y1 = -0x80000000
            for bel in r.bels:
                loc = ctx.getBelLocation(bel)
                bb.x0 = min(bb.x0, loc.x)
                bb.x1 = max(bb.x1, loc.x)
                bb.y0 = min(bb.y0, loc.y)
                bb.y1 = max(bb.y1, loc.y)
        else:
            bb.x0 = 0
            bb.y0 = 0
            bb.x1 = max_x
            bb.y1 = max_y
        constraint_region_bounds[r.name] = bb

# Build and solve in one direction
def build_solve_direction(yaxis, iteration):
    for i in range(5):
        esx = EquationSystem(len(solve_cells), len(solve_cells))
        build_equations(esx, yaxis, iteration)
        solve_equations(esx, yaxis)

# Check if a cell has any meaningful connectivity
def has_connectivity(cell):
    for x,  port in cell.ports:
        if port.net and port.net.driver.cell and port.net.users:
            return True

    return False

import random

# Build up a random initial placement, without regard to legality
# FIXME: Are there better approaches to the initial placement (e.g. greedy?)
def seed_placement():
    
    for bel in ctx.getBels():
        if ctx.checkBelAvail(bel):
            belType = ctx.getBelType(bel)
            availableBels.setdefault(belType, []).append(bel)

    for ty, bels in availableBels.items():
        random.shuffle(bels)

    for cell, cinfo in ctx.cells:
        if cinfo.bel:
            loc = ctx.getBelLocation(cinfo.bel)
            cell_locs[cell] = CellLoc(
                loc.x, loc.y, None, None, None, None,
                True,                    # locked
                ctx.getBelGlobalBuf(cinfo.bel) # global
            )
        elif not cinfo.constr_parent:
            belQueue = availableBels[cinfo.type]
            while True:
                bel = belQueue.pop()
                loc = ctx.getBelLocation(bel)
                cell_locs[cell] = CellLoc(
                    loc.x, loc.y, None, None, None, None,
                    True,                    # locked
                    ctx.getBelGlobalBuf(bel) # global
                )

                if cinfo.type not in ioBufTypes and has_connectivity(cinfo):
                    place_cells.append(cinfo)
                    break

                if ctx.isValidBelForCell(cinfo, bel):
                    ctx.bindBel(bel, cinfo, PlaceStrength.STRENGTH_STRONG)
                    cell_locs[cell].locked = True
                    break

                belQueue.insert(0, bel)

dont_solve = 0x7fffffff

def setup_solve_cells(celltypes=None):
    row = 0
    solve_cells[:] = []
    for cell, cinfo in ctx.cells:
        cinfo.udata = dont_solve
    for cell in place_cells:
        if celltypes is not None and cell.type not in celltypes:
            continue
        cell.udata = row
        row = row+1
        solve_cells.append(cell)

    for chained_first, chained_second in chain_root.items():
        ctx.cells[chained_first].udata = ctx.cells[chained_second.name].udata
    return row

UNCONSTR = -0x80000000

# Update the location of all children of a chain
def update_chain(cell, root):
    base = cell_locs[cell.name]
    for child in cell.constr_children:
        if child.name not in cell_locs:
            cell_locs[child.name] = CellLoc(
                0, 0, 0, 0, 0, 0, False, False
            )

        chain_size[root.name] += 1
        if child.constr_x != UNCONSTR:
            cell_locs[child.name].x = min(max_x, base.x + child.constr_x)
        else:
            cell_locs[child.name].x = base.x; # better handling of UNCONSTR?
        if child.constr_y != UNCONSTR:
            cell_locs[child.name].y = min(max_y, base.y + child.constr_y)
        else:
            cell_locs[child.name].y = base.y; # better handling of UNCONSTR?
        chain_root[child.name] = root
        if len(child.constr_children) > 0:
            update_chain(child, root)

def update_all_chains():
    for cell in place_cells:
        chain_size[cell.name] = 1
        if len(cell.constr_children) > 0:
            update_chain(cell, cell)

def foreach_port(net):
    if net.driver.cell:
        yield (net.driver, -1)
    for i, user in enumerate(net.users):
        yield (user, i)

# Build the system of equations for either X or Y
def build_equations(es, axis, iter=-1):
    # Return the x or y position of a cell, depending on ydir
    assert axis in ("x", "y")
    if axis == "x":
        cell_pos  = lambda cell: cell_locs[cell.name].x
        legal_pos = lambda cell: cell_locs[cell.name].legal_x
    else:
        cell_pos  = lambda cell: cell_locs[cell.name].y
        legal_pos = lambda cell: cell_locs[cell.name].legal_y

    es.reset()

    for net, ni in ctx.nets:
        if not ni.driver.cell:
            continue
        if len(ni.users) == 0:
            continue
        if cell_locs[ni.driver.cell.name].global_:
            continue
        # Find the bounds of the net in this axis, and the ports that correspond to these bounds
        lbport = None
        ubport = None
        lbpos =  0x7fffffff
        ubpos = -0x80000000
        for port, user_idx in foreach_port(ni):
            pos = cell_pos(port.cell)
            if pos < lbpos:
                lbpos = pos
                lbport = port

            if pos > ubpos:
                ubpos = pos
                ubport = port


        assert lbport
        assert ubport

        def stamp_equation(var, eqn, weight):
            if eqn.cell.udata == dont_solve:
                return
            row = eqn.cell.udata
            v_pos = cell_pos(var.cell)
            if var.cell.udata != dont_solve:
                es.add_coeff(row, var.cell.udata, weight)
            else:
                es.add_rhs(row, -v_pos * weight)
            if var.cell.name in cell_offsets:
                xoff, yoff = cell_offsets[var.cell.name]
                es.add_rhs(row, -weight * (xoff if axis=="x" else yoff))

        # Add all relevant connections to the matrix
        for port, user_idx in foreach_port(ni):
            this_pos = cell_pos(port.cell)
            for other in (lbport, ubport):
                if other == port:
                    continue

                o_pos = cell_pos(other.cell)
                weight = 1.0 / (len(ni.users) * max(1, abs(o_pos - this_pos)))

                if user_idx != -1 and ni.name in net_crit:
                    nc = net_crit[ni.name]
                    if user_idx < len(nc.criticality):
                        weight *= (1.0 + cfg.timingWeight *
                            (nc.criticality[user_idx] ** cfg.criticalityExponent))

                # If cell 0 is not fixed, it will stamp +w on its equation and -w on the other end's equation,
                # if the other end isn't fixed
                stamp_equation(port, port, weight)
                stamp_equation(port, other, -weight)
                stamp_equation(other, other, weight)
                stamp_equation(other, port, -weight)

    if iter != -1:
        alpha = cfg.alpha
        for row, solve_cell in enumerate(solve_cells):
            l_pos = legal_pos(solve_cell)
            c_pos = cell_pos(solve_cell)

            weight = alpha * iter / max(1, abs(l_pos - c_pos))
            # Add an arc from legalised to current position
            es.add_coeff(row, row, weight)
            es.add_rhs(row, weight * l_pos)


# Build the system of equations for either X or Y
def solve_equations(es, axis):
    assert axis in ("x", "y")
    if axis == "x":
        vals = [cell_locs[cell.name].x for cell in solve_cells]
    else:
        vals = [cell_locs[cell.name].y for cell in solve_cells]

    vals, what = es.solve(vals)

    for solve_cell, val in zip(solve_cells, vals):
        cell_loc = cell_locs[solve_cell.name]
        if axis == "y":
            cell_loc.rawy = val
            cell_loc.y = min(max_y, max(0, int(val)))
            if solve_cell.region:
                cell_loc.y = limit_to_reg_y(solve_cells.region, cell_loc.y)
        else:
            cell_loc.rawx = val
            cell_loc.x = min(max_x, max(0, int(val)))
            if solve_cell.region:
                cell_loc.x = limit_to_reg_x(solve_cells.region, cell_loc.x)

def total_hpwl():
    hpwl = 0
    for net, ni in ctx.nets:
        if not ni.driver.cell:
            continue
        drvloc = cell_locs[ni.driver.cell.name]
        if drvloc.global_:
            continue
        xmin = drvloc.x
        xmax = drvloc.x
        ymin = drvloc.y
        ymax = drvloc.y
        for user in ni.users:
            usrloc = cell_locs[user.cell.name]
            xmin = min(xmin, usrloc.x)
            xmax = max(xmax, usrloc.x)
            ymin = min(ymin, usrloc.y)
            ymax = max(ymax, usrloc.y)
        hpwl += (xmax - xmin) + (ymax - ymin)
    return hpwl

# Strict placement legalisation, performed after the initial HeAP spreading
def legalise_placement_strict(require_validity=False):
    global sl_time

    startt = time.time_ns()

    # Unbind all cells placed in this solution
    for cell, ci in ctx.cells:
        if ci.bel and (ci.udata != dont_solve or
                                   (ci.name in chain_root and chain_root[ci.name].udata != dont_solve)):
            ctx.unbindBel(ci.bel)

    # At the moment we don't follow the full HeAP algorithm using cuts for legalisation, instead using
    # the simple greedy largest-macro-first approach.
    remaining = []

    tie_breaker = 0
    for tie_breaker, cell in enumerate(solve_cells):
        remaining.append((-chain_size[cell.name], tie_breaker, cell))
    tie_breaker += 1

    heapq.heapify(remaining)

    ripup_radius = 2
    total_iters = 0
    total_iters_noreset = 0
    initial_count = len(remaining)
    while remaining:
        _, _, ci = heapq.heappop(remaining)

        # Was now placed, ignore
        if ci.bel:
            continue
        # log_info("   Legalising %s (%s)" % (ci.name, ci.type))
        bt, _ = bel_types[ci.type]
        fb = fast_bels[bt]
        radius = 0
        iter_ = 0
        iter_at_radius = 0
        placed = False
        bestBel = None
        best_inp_len = 0x7fffffff

        total_iters += 1
        total_iters_noreset += 1
        if total_iters > len(solve_cells):
            total_iters = 0
            ripup_radius = max(max(max_x, max_y), ripup_radius * 2)


        if (total_iters_noreset > max(5000, 8 * len(ctx.cells))):
            log_error("Unable to find legal placement for all cells, design is probably at utilisation limit.\n")


        while not placed:
            # Set a conservative timeout
            if (iter_ > max(10000, 3 * len(ctx.cells))):
                log_error("Unable to find legal placement for cell '%s', check constraints and utilisation.\n",
                          ctx.nameOf(ci))

            rx = radius
            ry = radius

            if ci.region:
                rx = min(radius, (constraint_region_bounds[ci.region.name].x1 -
                                       constraint_region_bounds[ci.region.name].x0) /
                                                      2 +
                                              1)
                ry = min(radius, (constraint_region_bounds[ci.region.name].y1 -
                                       constraint_region_bounds[ci.region.name].y0) /
                                                      2 +
                                              1)

            #ctx_rng = ctx.rng
            import random, math
            ctx_rng = lambda n: random.randrange(int(2**math.ceil(math.log2(n))))
            nx = ctx_rng(2 * rx + 1) + max(cell_locs[ci.name].x - rx, 0)
            ny = ctx_rng(2 * ry + 1) + max(cell_locs[ci.name].y - ry, 0)

            iter_ += 1
            iter_at_radius += 1
            if (iter_ >= (10 * (radius + 1))):
                def det_radius():
                    nonlocal radius
                    radius = min(max(max_x, max_y), radius + 1)
                    while (radius < max(max_x, max_y)):
                        for x in range(
                                max(0, cell_locs[ci.name].x - radius),
                                min(max_x, cell_locs[ci.name].x + radius)+1):
                            if (x >= len(fb)):
                                break
                            for y in range(
                                max(0, cell_locs[ci.name].y - radius),
                                min(max_y, cell_locs[ci.name].y + radius)+1):
                                if (y >= len(fb[x])):
                                    break
                                if len(fb[x][y]) > 0:
                                    return


                        radius = min(max(max_x, max_y), radius + 1)

                det_radius()
                iter_at_radius = 0
                iter_ = 0 # original code does this but it seems wrong

            if (nx < 0 or nx > max_x):
                continue
            if (ny < 0 or ny > max_y):
                continue

            # ny = nearest_row_with_bel[bt][ny]
            # nx = nearest_col_with_bel[bt][nx]

            if (nx >= len(fb)):
                continue
            if (ny >= len(fb[nx])):
                continue
            if len(fb[nx][ny]) == 0:
                continue

            need_to_explore = 2 * radius

            if iter_at_radius >= need_to_explore and bestBel:
                bound = ctx.getBoundBelCell(bestBel)
                if (bound != None):
                    ctx.unbindBel(bound.bel)
                    heapq.heappush(remaining,
                        (-chain_size[bound.name], tie_breaker, bound))
                    tie_breaker += 1

                ctx.bindBel(bestBel, ci, STRENGTH_WEAK)
                placed = True
                loc = ctx.getBelLocation(bestBel)
                cell_locs[ci.name].x = loc.x
                cell_locs[ci.name].y = loc.y
                break


            if len(ci.constr_children) == 0 and not ci.constr_abs_z:
                for sz in fb[nx][ny]:
                    if (ci.region != None and ci.region.constr_bels and not ci.region.sz in bels):
                        continue
                    if (ctx.checkBelAvail(sz) or (radius > ripup_radius or ctx_rng(20000) < 10)):
                        bound = ctx.getBoundBelCell(sz)
                        if (bound != None):
                            if (bound.constr_parent != None or bound.constr_children or
                                bound.constr_abs_z):
                                continue
                            ctx.unbindBel(bound.bel)

                        ctx.bindBel(sz, ci, STRENGTH_WEAK)
                        if (require_validity and not ctx.isBelLocationValid(sz)):
                            ctx.unbindBel(sz)
                            if (bound != None):
                                ctx.bindBel(sz, bound, STRENGTH_WEAK)
                        elif (iter_at_radius < need_to_explore):
                            ctx.unbindBel(sz)
                            if (bound != None):
                                ctx.bindBel(sz, bound, STRENGTH_WEAK)
                            input_len = 0
                            for port in ci.ports:
                                p = port.second
                                if (p.type != PORT_IN or p.net == None or p.net.driver.cell == None):
                                    continue
                                drv = p.net.driver.cell
                                if drv.name not in cell_locs:
                                    continue
                                drv_loc = cell_locs[drv.name]
                                if (drv_loc.global_):
                                    continue
                                input_len += abs(drv_loc.x - nx) + abs(drv_loc.y - ny)

                            if (input_len < best_inp_len):
                                best_inp_len = input_len
                                bestBel = sz

                            break
                        else:
                            assert ci.bel
                            if (bound != None):
                                heapq.heappush(remaining,
                                    (-chain_size[bound.name], tie_breaker, bound))
                                tie_breaker += 1
                            loc = ctx.getBelLocation(sz)
                            cell_locs[ci.name].x = loc.x
                            cell_locs[ci.name].y = loc.y
                            placed = True
                            break

            else:
                for sz in fb[nx][ny]:
                    loc = ctx.getBelLocation(sz)
                    if (ci.constr_abs_z and loc.z != ci.constr_z):
                        continue
                    #std::vector<std::pair<CellInfo *, BelId>> targets
                    #std::vector<std::pair<BelId, CellInfo *>> swaps_made
                    #std::queue<std::pair<CellInfo *, Loc>> visit
                    targets = []
                    swaps_made = []
                    visit = []
                    visit.append((ci, loc))
                    class PlaceFail(Exception): pass
                    visits = 0
                    try:
                        while visit:
                            visits += 1
                            vc, ploc = visit.pop(0)
                            assert not vc.bel
                            target = ctx.getBelByLocation(ploc)
                            if (vc.region != None and vc.region.constr_bels and not vc.region.target in bels):
                                raise PlaceFail()

                            if not target or ctx.getBelType(target) != vc.type:
                                raise PlaceFail()
                            bound = ctx.getBoundBelCell(target)
                            # Chains cannot overlap
                            if (bound != None):
                                if (bound.constr_z != UNCONSTR or bound.constr_parent != None or
                                    len(bound.constr_children) > 0 or bound.belStrength > STRENGTH_WEAK):
                                    raise PlaceFail()
                            targets.append((vc, target))
                            for child in vc.constr_children:
                                cloc = Loc()
                                cloc.x = ploc.x
                                cloc.y = ploc.y
                                cloc.z = ploc.z
                                if (child.constr_x != UNCONSTR):
                                    cloc.x += child.constr_x
                                if (child.constr_y != UNCONSTR):
                                    cloc.y += child.constr_y
                                if (child.constr_z != UNCONSTR):
                                    cloc.z = child.constr_z if child.constr_abs_z else (ploc.z + child.constr_z)
                                visit.append((child, cloc))

                        for xcell, target in targets:
                            bound = ctx.getBoundBelCell(target)
                            if (bound != None):
                                ctx.unbindBel(target)
                            ctx.bindBel(target, xcell, STRENGTH_STRONG)
                            swaps_made.append((target, bound))

                        for sm in swaps_made:
                            if not ctx.isBelLocationValid(sm[0]):
                                raise PlaceFail()

                    except PlaceFail:
                        for swap_first, swap_second in swaps_made:
                            ctx.unbindBel(swap_first)
                            if swap_second:
                                ctx.bindBel(swap_first, swap_second, STRENGTH_WEAK)

                        continue

                    for target_first, target_second in targets:
                        loc = ctx.getBelLocation(target_second)
                        cell_locs[target_first.name].x = loc.x
                        cell_locs[target_first.name].y = loc.y
                        #log_info("%s %d %d %d" % (target_first.name, loc.x, loc.y, loc.z))

                    for swap_first, swap_second in swaps_made:
                        if (swap_second != None):
                            # original code will implicitly insert new 0 entry into chain_size
                            heapq.heappush(remaining,
                                (-chain_size.get(swap_second.name, 0), tie_breaker, swap_second))
                            tie_breaker += 1

                    placed = True
                    break

        cell_name = ci.name
        assert isinstance(cell_name, str)
        assert placed
        assert ci.bel, cell_locs[cell_name]

    endt = time.time_ns()
    sl_time += duration(endt - startt)


def limit_to_reg_x(reg, val):
    if not reg:
        return val
    regname = reg.name
    limit_low = constraint_region_bounds[regname].x0
    limit_high = constraint_region_bounds[regname].x1
    return max(min(val, limit_high), limit_low)

def limit_to_reg_y(reg, val):
    if not reg:
        return val
    regname = reg.name
    limit_low = constraint_region_bounds[regname].y0
    limit_high = constraint_region_bounds[regname].y1
    return max(min(val, limit_high), limit_low)

class CutSpreader:
    def __init__(self, p, ty):
        type_idx, _ = bel_types[ty]
        # Context *ctx
        self.p = p                  # HeAPPlacer *p
        self.beltype = ty           # IdString self.beltype
        self.occupancy = []         # std::vector<std::vector<int>> occupancy
        self.groups = []            # std::vector<std::vector<int>> groups
        self.chaines = []           # std::vector<std::vector<ChainExtent>> chaines
        self.cell_extents = {}      # map<IdString, ChainExtent> cell_extents
        self.fb = fast_bels[type_idx] # std::vector<std::vector<std::vector<BelId>>> &fb

        self.regions = []           # std::vector<SpreaderRegion> regions
        self.merged_regions = set() # std::unordered_set<int> merged_regions

        # Cells at a location, sorted by real (not integer) x and y
        self.cells_at_location = [] # std::vector<std::vector<std::vector<CellInfo *>>> cells_at_location

    def run(self):
        startt = time.time_ns()
        self.init()
        self.find_overused_regions()
        for r in self.regions:
            if r.id in self.merged_regions:
                continue

            # log_info("%s (%d, %d) |_> (%d, %d) %d/%d\n", self.beltype, r.x0, r.y0, r.x1, r.y1, r.cells,
            #          r.bels)


        self.expand_regions()
        workqueue = [] # std::queue<std::pair<int, bool>> workqueue

        # orig = [] # std::vector<std::pair<double, double>> orig
        # if ctx.debug:
        #     for c in self.p.solve_cells:
        #         orig.emplace_back(self.p.cell_locs[c.name].rawx, self.p.cell_locs[c.name].rawy)

        for r in self.regions:
            if r.id in self.merged_regions:
                continue

            # log_info("%s (%d, %d) |_> (%d, %d) %d/%d\n", self.beltype, r.x0, r.y0, r.x1, r.y1, r.cells,
            #          r.bels)

            workqueue.append((r.id, False))
            # cut_region(r, false)

        while workqueue:
            rid, axis = workqueue.pop(0)
            r = self.regions[rid]
            if r.cells == 0:
                continue

            res = self.cut_region(r, axis)
            if res:
                workqueue.append((res[0], not axis))
                workqueue.append((res[1], not axis))
                continue

            # Try the other dir, in case stuck in one direction only
            res2 = self.cut_region(r, not axis)
            if res2:
                # log_info("RETRY SUCCESS\n")
                workqueue.append((res2[0], axis))
                workqueue.append((res2[1], axis))

        # if ctx.debug:
        #     with open("spread{}.csv".format(seq), "w") as sp:
        #         for (size_t i = 0; i < self.p.solve_cells.size(); i++):
        #             auto &c = self.p.solve_cells[i]
        #             if (c.type != self.beltype):
        #                 continue
        #             sp << orig[i].first << "," << orig[i].second << "," << self.p.cell_locs[c.name].rawx << "," << self.p.cell_locs[c.name].rawy << std::endl
        #
        #     with open("cells{}.csv".format(seq), "w") as oc:
        #         for (size_t y = 0; y <= self.p.max_y; y++):
        #             for (size_t x = 0; x <= self.p.max_x; x++):
        #                 oc << cells_at_location[x][y].size() << ", "
        #             oc << std::endl
        #
        #     seq += 1

        endt = time.time_ns()
        self.p.cl_time += duration(endt-startt)

    def occ_at(self, x, y):
        return self.occupancy[x][y]

    def bels_at(self, x, y):
        if x >= len(self.fb): return 0
        if y >= len(self.fb[x]): return 0
        return len(self.fb[x][y])

    def init(self):
        self.occupancy         = [[0]*(self.p.max_y+1) for _ in range(self.p.max_x + 1)]
        self.groups            = [[-1]*(self.p.max_y+1) for _ in range(self.p.max_x + 1)]
        self.chaines           = [[None]*(self.p.max_y+1) for _ in range(self.p.max_x + 1)]
        self.cells_at_location = [[[] for _ in range(self.p.max_y+1)] for _ in range(self.p.max_x + 1)]
        for x in range(self.p.max_x+1):
            for y in range(self.p.max_y+1):
                self.occupancy[x][y] = 0
                self.groups[x][y] = -1
                self.chaines[x][y] = ChainExtent(x, y, x, y)

        def set_chain_ext(cell, x, y):
            if cell not in self.cell_extents:
                self.cell_extents[cell] = ChainExtent(x, y, x, y)
            else:
                self.cell_extents[cell].x0 = min(self.cell_extents[cell].x0, x)
                self.cell_extents[cell].y0 = min(self.cell_extents[cell].y0, y)
                self.cell_extents[cell].x1 = max(self.cell_extents[cell].x1, x)
                self.cell_extents[cell].y1 = max(self.cell_extents[cell].y1, y)

        for cell, cell_loc in self.p.cell_locs.items():
            if ctx.cells[cell].type != self.beltype:
                continue
            if ctx.cells[cell].belStrength > STRENGTH_STRONG:
                continue
            self.occupancy[cell_loc.x][cell_loc.y] += 1
            # Compute ultimate extent of each chain root
            if cell in self.p.chain_root:
                set_chain_ext(self.p.chain_root[cell].name, cell_loc.x, cell_loc.y)
            elif ctx.cells[cell].constr_children:
                set_chain_ext(cell, cell_loc.x, cell_loc.y)

        for cell, cell_loc in self.p.cell_locs.items():
            if ctx.cells[cell].type != self.beltype:
                continue
            # Transfer chain extents to the actual chaines structure
            ce = None
            if cell in self.p.chain_root:
                ce = self.cell_extents[self.p.chain_root[cell].name] # ref
            elif ctx.cells[cell].constr_children:
                ce = self.cell_extents[cell] # ref
            if ce:
                lce = self.chaines[cell_loc.x][cell_loc.y]
                lce.x0 = min(lce.x0, ce.x0)
                lce.y0 = min(lce.y0, ce.y0)
                lce.x1 = max(lce.x1, ce.x1)
                lce.y1 = max(lce.y1, ce.y1)

        for cell in self.p.solve_cells:
            if cell.type != self.beltype:
                continue
            self.cells_at_location[self.p.cell_locs[cell.name].x][self.p.cell_locs[cell.name].y].append(cell)

    def merge_regions(self, merged, mergee):
        # Prevent grow_region from recursing while doing this
        for x in range(mergee.x0, mergee.x1+1):
            for y in range(mergee.y0, mergee.y1+1):
                # log_info("%d %d\n", groups[x][y], mergee.id)
                assert self.groups[x][y] == mergee.id
                self.groups[x][y] = merged.id
                merged.cells += self.occ_at(x, y)
                merged.bels += self.bels_at(x, y)

        self.merged_regions.add(mergee.id)
        self.grow_region(merged, mergee.x0, mergee.y0, mergee.x1, mergee.y1)

    def grow_region(self, r, x0, y0, x1, y1, init=False):
            # log_info("growing to (%d, %d) |_> (%d, %d)\n", x0, y0, x1, y1)
            if (x0 >= r.x0 and y0 >= r.y0 and x1 <= r.x1 and y1 <= r.y1) or init:
                return
            old_x0 = r.x0 + (1 if init else 0)
            old_y0 = r.y0
            old_x1 = r.x1
            old_y1 = r.y1
            r.x0 = min(r.x0, x0)
            r.y0 = min(r.y0, y0)
            r.x1 = max(r.x1, x1)
            r.y1 = max(r.y1, y1)

            def process_location(x, y):
                # Merge with any overlapping regions
                if self.groups[x][y] == -1:
                    r.bels += self.bels_at(x, y)
                    r.cells += self.occ_at(x, y)

                if self.groups[x][y] != -1 and self.groups[x][y] != r.id:
                    self.merge_regions(r, self.regions[self.groups[x][y]])
                self.groups[x][y] = r.id
                # Grow to cover any chains
                chaine = self.chaines[x][y]
                self.grow_region(r, chaine.x0, chaine.y0, chaine.x1, chaine.y1)

            for x in range(r.x0, old_x0):
                for y in range(r.y0, r.y1+1):
                    process_location(x, y)
            for x in range(old_x1 + 1, x1+1):
                for y in range(r.y0, r.y1+1):
                    process_location(x, y)
            for y in range(r.y0, old_y0):
                for x in range(r.x0, r.x1+1):
                    process_location(x, y)
            for y in range(old_y1 + 1, r.y1+1):
                for x in range(r.x0, r.x1+1):
                    process_location(x, y)

    def find_overused_regions(self):
        for x in range(0, self.p.max_x+1):
            for y in range(0, self.p.max_y+1):
                # Either already in a group, or not overutilised. Ignore
                if self.groups[x][y] != -1 or (self.occ_at(x, y) <= self.bels_at(x, y)):
                    continue
                # log_info("%d %d %d\n", x, y, occ_at(x, y))
                id_ = len(self.regions)
                self.groups[x][y] = id_
                reg = SpreaderRegion()
                reg.id = id_
                reg.x0 = reg.x1 = x
                reg.y0 = reg.y1 = y
                reg.bels = self.bels_at(x, y)
                reg.cells = self.occ_at(x, y)
                # Make sure we cover carries, etc
                self.grow_region(reg, reg.x0, reg.y0, reg.x1, reg.y1, True)

                expanded = True
                while expanded:
                    expanded = False
                    # Keep trying expansion in x and y, until we find no over-occupancy cells
                    # or hit grouped cells

                    # First try expanding in x
                    if reg.x1 < self.p.max_x:
                        over_occ_x = False
                        for y1 in range(reg.y0, reg.y1+1):
                            if self.occ_at(reg.x1 + 1, y1) > self.bels_at(reg.x1 + 1, y1):
                                # log_info("(%d, %d) occ %d bels %d\n", reg.x1+ 1, y1, occ_at(reg.x1 + 1, y1),
                                # bels_at(reg.x1 + 1, y1))
                                over_occ_x = True
                                break

                        if over_occ_x:
                            expanded = True
                            self.grow_region(reg, reg.x0, reg.y0, reg.x1 + 1, reg.y1)

                    if reg.y1 < self.p.max_y:
                        over_occ_y = False
                        for x1 in range(reg.x0, reg.x1+1):
                            if self.occ_at(x1, reg.y1 + 1) > self.bels_at(x1, reg.y1 + 1):
                                # log_info("(%d, %d) occ %d bels %d\n", x1, reg.y1 + 1, occ_at(x1, reg.y1 + 1),
                                # bels_at(x1, reg.y1 + 1))
                                over_occ_y = True
                                break
                        if over_occ_y:
                            expanded = True
                            self.grow_region(reg, reg.x0, reg.y0, reg.x1, reg.y1 + 1)
                self.regions.append(reg)

    def expand_regions(self):
        overu_regions = [r.id for r in self.regions if r.overused()]

        while overu_regions:
            rid = overu_regions.pop(0)
            if rid in self.merged_regions:
                continue
            reg = self.regions[rid]
            while reg.overused():
                changed = False
                if reg.x0 > 0:
                    self.grow_region(reg, reg.x0 - 1, reg.y0, reg.x1, reg.y1)
                    changed = True
                    if not reg.overused():
                        break

                if reg.x1 < self.p.max_x:
                    self.grow_region(reg, reg.x0, reg.y0, reg.x1 + 1, reg.y1)
                    changed = True
                    if not reg.overused():
                        break

                if reg.y0 > 0:
                    self.grow_region(reg, reg.x0, reg.y0 - 1, reg.x1, reg.y1)
                    changed = True
                    if not reg.overused():
                        break

                if reg.y1 < self.p.max_y:
                    self.grow_region(reg, reg.x0, reg.y0, reg.x1, reg.y1 + 1)
                    changed = True
                    if not reg.overused():
                        break

                if not changed:
                    if reg.cells > reg.bels:
                        log_error("Failed to expand region (%d, %d) |_> (%d, %d) of %d %ss" %
                            (reg.x0, reg.y0, reg.x1, reg.y1, reg.cells, self.beltype))
                    else:
                        break

    def cut_region(self, r, dir):
        p = self.p
        cut_cells = self.cut_cells = []
        cal = self.cells_at_location
        total_cells = 0
        total_bels = 0
        for x in range(r.x0, r.x1+1):
            for y in range(r.y0, r.y1+1):
                cut_cells.extend(cal[x][y])
                total_bels += self.bels_at(x, y)

        for cell in cut_cells:
            total_cells += self.p.chain_size.get(cell.name, 1)

        if dir:
            cut_cells.sort(key=lambda cinfo: self.p.cell_locs[cinfo.name].rawy)
        else:
            cut_cells.sort(key=lambda cinfo: self.p.cell_locs[cinfo.name].rawx)

        if len(cut_cells) < 2:
            return {}
        # Find the cells midpoint, counting chains in terms of their total size - making the initial source cut
        pivot_cells = 0
        pivot = 0
        for cell in cut_cells:
            pivot_cells += self.p.chain_size.get(cell.name, 1)
            if (pivot_cells >= total_cells / 2):
                break
            pivot+=1

        if (pivot == len(cut_cells)):
            pivot = len(cut_cells) - 1
        # log_info("orig pivot %d lc %d rc %d\n", pivot, pivot_cells, r.cells - pivot_cells)

        # Find the clearance required either side of the pivot
        clearance_l = 0
        clearance_r = 0
        for i in range(len(cut_cells)):
            if cut_cells[i].name in self.cell_extents:
                ce = self.cell_extents[cut_cells[i].name]
                size = (ce.y1 - ce.y0 + 1) if dir else (ce.x1 - ce.x0 + 1)
            else:
                size = 1

            if int(i) < pivot:
                clearance_l = max(clearance_l, size)
            else:
                clearance_r = max(clearance_r, size)

        # Find the target cut that minimises difference in utilisation, whilst trying to ensure that all chains
        # still fit

        # First trim the boundaries of the region in the axis-of-interest, skipping any rows/cols without any
        # bels of the appropriate type
        trimmed_l = r.y0 if dir else r.x0
        trimmed_r = r.y1 if dir else r.x1
        while (trimmed_l < (r.y1 if dir else r.x1)):
            have_bels = False
            for i in range(r.x0 if dir else r.y0, (r.x1 if dir else r.y1)+1):
                if self.bels_at(i if dir else trimmed_l, trimmed_l if dir else i) > 0:
                    have_bels = True
                    break

            if have_bels:
                break
            trimmed_l += 1

        while (trimmed_r > (r.y0 if dir else r.x0)):
            have_bels = False
            for i in range(r.x0 if dir else r.y0, (r.x1 if dir else r.y1)+1):
                if self.bels_at(i if dir else trimmed_r, trimmed_r if dir else i) > 0:
                    have_bels = True
                    break

            if have_bels:
                break
            trimmed_r -= 1

        # log_info("tl %d tr %d cl %d cr %d\n", trimmed_l, trimmed_r, clearance_l, clearance_r)
        if ((trimmed_r - trimmed_l + 1) <= max(clearance_l, clearance_r)):
            return {}
        # Now find the initial target cut that minimises utilisation imbalance, whilst
        # meeting the clearance requirements for any large macros
        left_cells = pivot_cells
        right_cells = total_cells - pivot_cells
        left_bels = 0
        right_bels = total_bels
        best_tgt_cut = -1
        best_deltaU = max_double
        target_cut_bels = (None, None)
        for i in range(trimmed_l, trimmed_r+1):
            slither_bels = 0
            for j in range(r.x0 if dir else r.y0, (r.x1 if dir else r.y1)+1):
                slither_bels += self.bels_at(j, i) if dir else self.bels_at(i, j)

            left_bels += slither_bels
            right_bels -= slither_bels
            if (((i - trimmed_l) + 1) >= clearance_l and ((trimmed_r - i) + 1) >= clearance_r):
                # Solution is potentially valid
                if left_bels == 0 or right_bels == 0: continue  # not in original code
                aU = abs(float(left_cells) / float(left_bels) - float(right_cells) / float(right_bels))
                if aU < best_deltaU:
                    best_deltaU = aU
                    best_tgt_cut = i
                    target_cut_bels = (left_bels, right_bels)

        if best_tgt_cut == -1:
            return {}
        left_bels, right_bels = target_cut_bels
        # log_info("pivot %d target cut %d lc %d lb %d rc %d rb %d\n", pivot, best_tgt_cut, left_cells, left_bels,
        # right_cells, right_bels)

        # Peturb the source cut to eliminate overutilisation
        while (pivot > 0 and (float(left_cells) / float(left_bels) > float(right_cells) / float(right_bels))):
            move_cell = cut_cells[pivot]
            size = p.chain_size.get(move_cell.name, 1)
            left_cells -= size
            right_cells += size
            pivot -= 1

        while (pivot < len(cut_cells) - 1 and
               (float(left_cells) / float(left_bels) < float(right_cells) / float(right_bels))):
            move_cell = cut_cells[pivot + 1]
            size = p.chain_size.get(move_cell.name, 1)
            left_cells += size
            right_cells -= size
            pivot += 1

        # log_info("peturbed pivot %d lc %d lb %d rc %d rb %d\n", pivot, left_cells, left_bels, right_cells,
        # right_bels)
        # Split regions into bins, and then spread cells by linear interpolation within those bins
        def spread_binlerp(cells_start, cells_end, area_l, area_r):
            N = cells_end - cells_start
            if N <= 2:
                for i in range(cells_start, cells_end):
                    pos = area_l + i * ((area_r - area_l) / N)
                    if dir:
                        p.cell_locs[cut_cells[i].name].rawy = pos
                    else:
                        p.cell_locs[cut_cells[i].name].rawx = pos

                return

            # Split region into up to 10 (K) bins
            K = min(N, 10)
            bin_bounds = [] # [(cell start, area start)] # std::vector<std::pair<int, double>>
            bin_bounds.append((cells_start, area_l))
            for i in range(1, K):
                bin_bounds.append((cells_start + int((N * i) / K), area_l + ((area_r - area_l + 0.99) * i) / K))
            bin_bounds.append((cells_end, area_r + 0.99))
            for i in range(K):
                bl = bin_bounds[i]
                br = bin_bounds[i + 1]

                if dir:
                    orig_left = p.cell_locs[cut_cells[bl[0]].name].rawy
                else:
                    orig_left = p.cell_locs[cut_cells[bl[0]].name].rawx

                if dir:
                    orig_right = p.cell_locs[cut_cells[br[0] - 1].name].rawy
                else:
                    orig_right = p.cell_locs[cut_cells[br[0] - 1].name].rawx

                m = (br[1] - bl[1]) / max(0.00001, orig_right - orig_left)
                for j in range(bl[0], br[0]):
                    cr = cut_cells[j].region
                    if cr is not None:
                        # Limit spreading bounds to constraint region; if applicable
                        brsc = p.limit_to_reg(cr, br[1], dir)
                        blsc = p.limit_to_reg(cr, bl[1], dir)
                        mr = (brsc - blsc) / max(0.00001, orig_right - orig_left)
                        if dir:
                            pos = p.cell_locs[cut_cells[j].name].rawy
                            assert pos >= orig_left and pos <= orig_right
                            pos = blsc + mr * (pos - orig_left)
                            p.cell_locs[cut_cells[j].name].rawy = pos
                        else:
                            pos = p.cell_locs[cut_cells[j].name].rawx
                            assert pos >= orig_left and pos <= orig_right
                            pos = blsc + mr * (pos - orig_left)
                            p.cell_locs[cut_cells[j].name].rawx = pos
                    else:
                        if dir:
                            pos = p.cell_locs[cut_cells[j].name].rawy
                            assert pos >= orig_left and pos <= orig_right
                            pos = bl[1] + m * (pos - orig_left)
                            p.cell_locs[cut_cells[j].name].rawy = pos
                        else:
                            pos = p.cell_locs[cut_cells[j].name].rawx
                            assert pos >= orig_left and pos <= orig_right
                            pos = bl[1] + m * (pos - orig_left)
                            p.cell_locs[cut_cells[j].name].rawx = pos

                    # log("[%f, %f] . [%f, %f]: %f . %f\n", orig_left, orig_right, bl[1], br[1],
                    # orig_pos, pos)

        spread_binlerp(0, pivot + 1, trimmed_l, best_tgt_cut)
        spread_binlerp(pivot + 1, len(cut_cells), best_tgt_cut + 1, trimmed_r)
        # Update various data structures
        for x in range(r.x0, r.x1+1):
            for y in range(r.y0, r.y1+1):
                self.cells_at_location[x][y].clear()

        for cell in cut_cells:
            cl = p.cell_locs[cell.name]
            cl.x = min(r.x1, max(r.x0, int(cl.rawx)))
            cl.y = min(r.y1, max(r.y0, int(cl.rawy)))
            self.cells_at_location[cl.x][cl.y].append(cell)
            # log_info("spread pos %d %d\n", cl.x, cl.y)

        rl = SpreaderRegion()
        rr = SpreaderRegion()
        rl.id = len(self.regions)
        rl.x0 = r.x0
        rl.y0 = r.y0
        rl.x1 = r.x1 if dir else best_tgt_cut
        rl.y1 = best_tgt_cut if dir else r.y1
        rl.cells = left_cells
        rl.bels = left_bels
        rr.id = len(self.regions) + 1
        rr.x0 = r.x0 if dir else (best_tgt_cut + 1)
        rr.y0 = (best_tgt_cut + 1) if dir else r.y0
        rr.x1 = r.x1
        rr.y1 = r.y1
        rr.cells = right_cells
        rr.bels = right_bels
        self.regions.append(rl)
        self.regions.append(rr)
        for x in range(rl.x0, rl.x1+1):
            for y in range(rl.y0, rl.y1+1):
                self.groups[x][y] = rl.id
        for x in range(rr.x0, rr.x1+1):
            for y in range(rr.y0, rr.y1+1):
                self.groups[x][y] = rr.id
        return (rl.id, rr.id)

place()
#code.interact(local=locals())
