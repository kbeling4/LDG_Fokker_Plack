import grid
import particle as prt
import material as mat


def main():
    Emin = 1
    Emax = 10
    Ne = 10

    Xmin = 0
    Xmax = 10
    Nx = 10

    particle = prt.Particle()
    material = mat.Material()
    
    g = grid.Grid(Ne, Emin, Emax, Nx, Xmin, Xmax)
    Enodes = g.get_EnodesUniform()
    Egrid = g.get_Egrid()
    Xgrid = g.get_Xgrid()
    Agi = g.get_Agi( 2, 2, particle, material )
    print( Xgrid )

if __name__ == "__main__": main()
