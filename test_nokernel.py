import cupy as cp
from libgputrace import trace_trajectory


EARTH_DIPOLE_B0 = -30e3   # nT

    
    
def main():
    """Main method of the program."""
    # Setup axes and grid
    x_axis = cp.arange(-10, 10, grid_spacing)
    y_axis = cp.arange(-10, 10, grid_spacing)
    z_axis = cp.arange(-5, 5, grid_spacing)
    axes = Axes(x_axis, y_axis, z_axis)
    
    x_grid, y_grid, z_grid = cp.meshgrid(
        axes.x, axes.y, axes.z,
        indexing='ij'
    )

    print('Grid Shape:', x_grid.shape)
    
    # Calculate Dipole on Grid
    r_grid = cp.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

    Bx = 3 * x_grid * z_grid * EARTH_DIPOLE_B0 / r_grid**5
    By = 3 * y_grid * z_grid * EARTH_DIPOLE_B0 / r_grid**5
    Bz = (3 * z_grid**2 - r_grid**2) * EARTH_DIPOLE_B0 / r_grid**5
    Btot = cp.sqrt(Bx**2 + By**2 + Bz**2)

    Ex = cp.zeros(Bx.shape)
    Ey = cp.zeros(Bx.shape)
    Ez = cp.zeros(Bx.shape)
    
    # Instantiate particle positions and parallel velocity
    pos_x = cp.arange(3, 8, .01)
    pos_y = cp.zeros(pos_x.shape)
    pos_z = cp.zeros(pos_x.shape)

    pos = Positions(pos_x, pos_y, pos_z)
    print('Number of particles:', pos_x.size)

    vpar = vpar_start * cp.ones(pos.x.shape)    

    # History of positions
    hist = PositionHistory(
        x=cp.zeros((MAX_ITERS, pos.x.size)),
        y=cp.zeros((MAX_ITERS, pos.y.size)),
        z=cp.zeros((MAX_ITERS, pos.y.size))
    )

    print('Iterations:', MAX_ITERS)
    
    start_time = time.time()
    sim(pos, vpar, hist, Bx, By, Bz, Btot, Ex, Ey, Ez, axes)
    end_time = time.time()
    
    print('time = ', end_time - start_time, 's')

    # Collect history
    hist_x = hist.x.get()
    hist_y = hist.y.get()
    hist_z = hist.z.get()

    # Write output for visualization
    d = {}
    
    for i in range(0, pos.x.size, 50):
        d[f'x{i}'] = hist_x[:, i]
        d[f'y{i}'] = hist_y[:, i]
        d[f'z{i}'] = hist_z[:, i]        
    
    pd.DataFrame(d).to_csv('out.csv')

    
#@line_profiler.profile


if __name__ == '__main__':
    main()
