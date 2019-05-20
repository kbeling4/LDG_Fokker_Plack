import numpy as np
from scipy import special
import physics as phy

class Grid:
    def __init__(self, Ne, Emin, Emax, Nx, Xmin, Xmax):
        self.Ne     = Ne
        self.Nx     = Nx
        self.Emin   = Emin
        self.Emax   = Emax
        self.Xmin   = Xmin
        self.Xmax   = Xmax
        self.Enodes = np.zeros( Ne )
        self.Xnodes = np.linspace( Xmin, Xmax, self.Nx, endpoint=True)
        self.Egrid  = np.array( np.zeros( ( self.Ne+1, self.Nx+1 ) ) )
        self.Xgrid  = np.array( np.zeros( ( self.Ne+1, self.Nx+1 ) ) )
        self.Agi = np.array( np.zeros( ( 8, 8 ) ) )
        self.Phi_b = np.array( np.zeros( ( self.Ne+1, self.Nx+1 ) ) )
        self.R_b = np.array( np.zeros( ( self.Ne+1, self.Nx+1 ) ) )
        
    def get_EnodesUniform( self ):
        self.Enodes[0]         = self.Emin
        self.Enodes[self.Ne-1] = self.Emax
        for i in range( 2, self.Ne ):
            self.Enodes[i-1] = self.Emin + ((i - 1)/( self.Ne - 1))*( self.Emax - self.Emin )
        return self.Enodes

    def get_Egrid( self ):
        del_E = self.Enodes[1] - self.Enodes[0]
        Ebounds = np.linspace( self.Emin-del_E/2, self.Emax+del_E/2, self.Ne+1 );
        for i in range( 0, self.Ne+1 ):
            for j in range( 0, self.Ne+1 ):
                self.Egrid[j,i] = Ebounds[j]
        return self.Egrid

    def get_Xgrid( self ):
        del_X = self.Xnodes[1] - self.Xnodes[0]
        Xbounds = np.linspace( self.Xmin-del_X/2, self.Xmax+del_X/2, self.Nx+1 );
        for i in range( 0, self.Nx+1 ):
            for j in range( 0, self.Nx+1 ):
                self.Xgrid[i,j] = Xbounds[j]
        return self.Xgrid

    def get_Agi( self, g, i, particle, material ):
        physics = phy.Physics()
        del_x = self.Xnodes[i] - self.Xnodes[i-1]
        del_E = self.Enodes[g] - self.Enodes[g-1]
        alpha = physics.S( self.Enodes[g], particle, material )
                    # + physics.dT( self.Enodes[g], particle, material ) )/ del_E - 2*(
                    #     physics.dS( self.Enodes[g], particle, material )
                    #     + 0.5*physics.ddT( self.Enodes[g], particle, material ) ) 
        beta = 3*( physics.S( self.Enodes[g], particle, material )
                    + physics.dT( self.Enodes[g], particle, material ))/ del_E - (
                        physics.dS( self.Enodes[g], particle, material )
                        + 0.5*physics.ddT( self.Enodes[g], particle, material ) )
        gamma = 3*( 0.5*physics.T( self.Enodes[g], particle, material ) ) / del_E
        delta = 3*( physics.S( self.Enodes[g], particle, material )
                   + physics.dT( self.Enodes[g], particle, material ))/ del_E - (
                       physics.dS( self.Enodes[g], particle, material )
                       + 0.5*physics.ddT( self.Enodes[g], particle, material ) )
        eps = 3*( physics.S( self.Enodes[g], particle, material )
                   + physics.dT( self.Enodes[g], particle, material ))/ del_E + 2*(
                       physics.dS( self.Enodes[g], particle, material )
                       + 0.5*physics.ddT( self.Enodes[g], particle, material ) )
        # Row 1
        self.Agi[0,0] = 1 + del_x*alpha/3
        self.Agi[0,1] = del_x*alpha/6 - 0.5
        self.Agi[0,2] = 0.5 + del_x*beta/3
        self.Agi[0,3] = del_x*beta/6 - 0.5
        self.Agi[0,4] = del_x*gamma/3
        self.Agi[0,5] = del_x*gamma/6
        self.Agi[0,6] = del_x*gamma/3
        self.Agi[0,7] = del_x*gamma/6
        # Row 2
        self.Agi[1,0] = del_x/del_E
        self.Agi[1,1] = del_x/(2*del_E)
        self.Agi[1,2] = del_x/del_E
        self.Agi[1,3] = del_x/(2*del_E)
        self.Agi[1,4] = (2/3)*del_x
        self.Agi[1,5] = del_x/3
        self.Agi[1,6] = del_x/3
        self.Agi[1,7] = del_x/6
        # Row 3
        self.Agi[2,0] = 1 + del_x*alpha/6
        self.Agi[2,1] = del_x*alpha/3 + 1
        self.Agi[2,2] = 0.5 + del_x*beta/6
        self.Agi[2,3] = del_x*beta/3 + 0.5
        self.Agi[2,4] = del_x*gamma/6
        self.Agi[2,5] = del_x*gamma/3
        self.Agi[2,6] = del_x*gamma/6
        self.Agi[2,7] = del_x*gamma/3
        # Row 4
        self.Agi[3,0] = del_x/(2*del_E)
        self.Agi[3,1] = del_x/del_E
        self.Agi[3,2] = del_x/(2*del_E)
        self.Agi[3,3] = del_x/del_E
        self.Agi[3,4] = del_x/3
        self.Agi[3,5] = (2/3)*del_x
        self.Agi[3,6] = del_x/6
        self.Agi[3,7] = del_x/3
        # Row 5
        self.Agi[4,0] = 0.5 + del_x*delta/3
        self.Agi[4,1] = del_x*delta/6 - 0.5
        self.Agi[4,2] = 1 - del_x*eps/3
        self.Agi[4,3] = (-1)*del_x*eps/6 - 1
        self.Agi[4,4] = del_x*gamma/3
        self.Agi[4,5] = del_x*gamma/6
        self.Agi[4,6] = (-1)*del_x*gamma/3
        self.Agi[4,7] = (-1)*del_x*gamma/6
        # Row 6
        self.Agi[5,0] = del_x/del_E
        self.Agi[5,1] = del_x/(2*del_E)
        self.Agi[5,2] = (-1)*del_x/del_E
        self.Agi[5,3] = (-1)*del_x/(2*del_E)
        self.Agi[5,4] = del_x/3
        self.Agi[5,5] = del_x/6
        self.Agi[5,6] = (2/3)*del_x
        self.Agi[5,7] = del_x/3
        # Row 7
        self.Agi[6,0] = 0.5 + del_x*delta/6
        self.Agi[6,1] = 0.5 + del_x*delta/3
        self.Agi[6,2] = 1 - del_x*eps/6
        self.Agi[6,3] = del_x*eps/3 + 1
        self.Agi[6,4] = del_x*gamma/6
        self.Agi[6,5] = del_x*gamma/3
        self.Agi[6,6] = (-1)*del_x*gamma/6
        self.Agi[6,7] = (-1)*del_x*gamma/3
        # Row 8
        self.Agi[7,0] = del_x/(2*del_E)
        self.Agi[7,1] = del_x/del_E
        self.Agi[7,2] = (-1)*del_x/(2*del_E)
        self.Agi[7,3] = (-1)*del_x/del_E
        self.Agi[7,4] = del_x/6
        self.Agi[7,5] = del_x/3
        self.Agi[7,6] = del_x/3
        self.Agi[7,7] = (2/3)*del_x
        
        return self.Agi
