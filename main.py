# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:25:02 2016

@author: Eric Schmidt
"""

'''
Written for work on Brookhavel National Lab's G-2 experiment.

Note on coordinates: Local coordinates are usually referred to where the
muon position is x,y,s where x=0, y=0 are the optimum path position and s is
the position around the ring. However, the code runs using global coordinates
where the center of the ring is the origin and z is 'up/down' So when talking
about the phase space, y-y' is used, which are in local coordinates. The
global coordinates used for the calculations are not refered to in any
qualitative sense.

The code tracks how a muon's y-y' phase space (in local coordinates) is
affected by small drops in the voltage supplied to the bottom plate in the
double quads. The change can either be a kick of +180 V or a drop of -180 V.
The change occurs in about 156 ns and the recovery is exponential with a time
constant of 5 micro seconds.

Assumptions include:

    - x-position (in local coordinates) is always at optimum position (x=0)
    - momentum is always at the 'magic' momentum
    - there are no edge effects at the edges of the quads, either the E-field
        is 'on' or it's 'off', the change is assumed to occur faster than a
        single time-step

The file 'callable_functions.py' includes miscellaneous functions used by this
code.
    
'''


import numpy as np
import matplotlib.pyplot as plt
import callable_functions as cf

def main():
    
    ''' Editable '''
    
    make_plots = 1                      # 1 to make plots, 0 to skip
    save_plots = 1                      # 1 to save plots, 0 to skip
    vary = 1                            # 1 to include the change in E-field
    plot_save_text = "ttfull_quad_entire" # Text to distinguish specific saves
    full_quad_coverage = 1              # 1 to place quads around entire ring
    yprime_limit = 0.00001              # To record y_max
#    lw = 0.1                            # Thickness of plotted lines
    steps = 5*10**5                     # Number of steps for integration
    dt = 10**-10                        # Timestep for integration
    d_i = int(1010)
    
    ''' Less Editable (but can be if the change is well understood) '''    
    
    tt = np.linspace(0,dt*steps,steps)  # For plotting
    tt = np.reshape(tt,(steps,1))
    p_mag = 3.09435*10**9               # (eV/c) Possible positron momentums
    q = 1                               # (e) Positron charge
    c = 2.99792458*10**8                # (m/s) Speed of light
    m = 105.65837*10**6                 # (eV/c**2) Muon mass
    B = np.array([0,0,1.4513])          # (T) Magnetic field
    R = 7.112                           # (m) Radius of ideal orbit
    dur = 1.56*10**-7                   # (s) The time it takes for the kick
    theta = 0                           # (rads) Initial theta in global coords
    n = .142                            # () Used in E-field
    recovered = -2                      # 0 if E-field not recovered from drop
    if vary == 1:
        drop = -180                      # Drop in voltage
    elif vary == 0:
        drop = 0
        plot_save_text = "no_vary"      # Text to distinguish specific saves
    tc = 5*10**-6                       # (s) Recovery time constant
    
    if drop > 0:
        save_folder = "Drop/"
    elif drop < 0:
        save_folder = "Kick/"
    else:
        save_folder = ""
    
    r_i = 0
    
#==============================================================================
#   Create quad geometries
#==============================================================================
    
    ''' Single-quad electrodes '''
    
    sqel_theta_base = 90 - np.array([31.89,44.89])-4
    sqel_theta = 90 - np.array([31.89,44.89])-4
    
    i = 1
    while i < 4:
        sqel_theta = np.row_stack((
            sqel_theta, sqel_theta_base + i*90
        ))
        i = i + 1
        
    sqel_theta = sqel_theta*np.pi/180
    
    ''' Double-quad electrodes '''
    
    dqel_theta_base = 90 - np.array([48.89,74.89])-4
    dqel_theta = 90 - np.array([48.89,74.89])-4
    
    i = 1
    while i < 4:
        dqel_theta = np.row_stack((
            dqel_theta, dqel_theta_base + i*90
        ))
        i = i + 1
        
    dqel_theta = dqel_theta*np.pi/180  
    
#==============================================================================
#   Initialize misc. variables
#==============================================================================
    
    # Set x,y,z positions
    
    x = np.zeros((3))             # Initialize positon array
    x[0] = R*np.cos(theta)            # (m) x-position
    x[1] = R*np.sin(theta)            # (m) y-position
    x[2] = -40*10**-3                 # (m) z-positio
    
    yprime = 0
    
    # Positron momentum at decay
#    v = np.zeros((steps,3))             # (m/s) Initialize velocity
    
    # (eV/c) Momentum
    p = cf.getPositronMomentumAtDecay(theta,p_mag)
    
    beta = cf.momentum2Beta(p,m)        # () Relativistic beta
    v = beta*c                          # (m/s) Initial velcoty
    s = dt * np.sqrt(v[0]**2 + v[1]**2) # (m) Initial distance traveled
    
    beta_mag = cf.mag(beta)             # Magnitude of relativistic beta vector
       
    gamma = cf.beta2Gamma(beta_mag)     # Relativistic gamma 
    
    ## Electric field

    ''' In the loop below, E is determined by the position of the particle and
        E_tot is E + the kick/drop if applicable and is what will be used to
        determine the force of the next loop iteration. E0 is the maximum
        strength of the E-field used to plot at the end.'''
    
    E = cf.getElectricField(x,R,B,n)            # Initial electric field values
    
    E_tot = cf.getElectricField(x,R,B,n)        # Initial electric field values
    
    E0 = np.zeros((steps))
    E0[0] = cf.getElectricField(x,R,B,n)[2]     # Set initial electric field
    
    E0_max = cf.getElectricField(x,R,B,n)[2]    # Set initial electric field
    
    E_drop = np.array([0,0,drop])/0.1           # E-field drop strength
    drop_slope = E_drop/(dur)                   # E-field drop rate
    
    # Force vector
    
    F = cf.forceDueFields(v,B,E,q)              # Initial force due fields
    
    y_max = np.zeros((int(steps),1))            # (m) Initialize
    ymaxc = 0                                   # y_max counter
    y_min = np.zeros((int(steps),1))            # (m) Initialize
    yminc = 0                                   # y_max counter
    
    # Used to note the furthest from 0 the muon ever reaches
    x_max = np.zeros((2))                       # x[0] = -y, x[1] = +y
    
    # Main loop counter
    i = 0
    
    # Counter for E-field exponential recovery
    k = 0
    
    # Counter for E-field drop
    l = 0
    
    ''' RK4 work '''
    
    while i < steps - 1:
        
        # Relativistic mass to find the acceleration from the force
        rmass = m*gamma/c**2        # (kg)
        
        a = F/(rmass)               # (m/s^-2) Acceleration
        dv1 = dt * a
        
        a = cf.forceDueFields(v + dv1/2,B,E_tot,q)/(rmass)
        dv2 = dt * a
        
        a = cf.forceDueFields(v + dv2/2,B,E_tot,q)/(rmass)
        dv3 = dt * a
        
        a = cf.forceDueFields(v + dv3,B,E_tot,q)/(rmass)
        dv4 = dt * a
        
        # New velocity vector
        
        v = v + (dv1 + 2*dv2 + 2*dv3 + dv4) / 6
        
        x_old = x                   # x_old will be used later
        
        x = x_old + dt * v          # New position vector
        
        # Check if the current position is further from 0 than before and
        # replace x_max if it is.        
        
        if x[2] < 0 and x[2] < x_max[0]:
            x_max[0] = x[2]
        if x[2] > 0 and x[2] > x_max[1]:
            x_max[1] = x[2]
        
        gamma = cf.beta2Gamma(cf.mag(v)/c)
        
        # Get the electric field based on position
        E = cf.getElectricField(x,R,B,n)
        
        # New arc position
        
        s_old = s
        s = s_old + dt * np.sqrt(v[0]**2 + v[1]**2)
        
        yprime = (x[2] - x_old[2])/(s - s_old) # (dy/ds)
        
        # If the drop/kick has not occured or if E-field has recovered to
        # within preset limits, or if drop/kick will never occur (vary == 0)        
        
        if recovered == -2 or recovered == 1:
            
            # Update the maximum E-field for plotting
            E0[i+1] = E0_max
            
            # Check if inside a double quad
            if cf.passthroughElementContact(x,dqel_theta):
                E_tot = E
        
            # Check if inside a single quad
            elif cf.passthroughElementContact(x,sqel_theta):
                E_tot = E

            # If particle is outside the quads
            else: 

                if full_quad_coverage == 1:
                    E_tot = E
                elif full_quad_coverage == 0:
                    E_tot = 0
            
        # If the drop/kick is set to occur
        if vary == 1:
        
            ''' Change this if statement to cause the voltage kick/drop to
            occur at different points around y,y' '''
        
            if i > d_i and x[2] < 0 and np.abs(yprime) < 0.00001 and recovered == -2:
                recovered = -1
                
            # During the actual kick/drop phase
            if recovered == -1:
                
                E0[i+1] = E0[i] - (drop_slope[2])*dt
                        
                if cf.passthroughElementContact(x,dqel_theta):
                    E_tot = E - (drop_slope)*dt*l
            
                elif cf.passthroughElementContact(x,sqel_theta):
                    E_tot = E

                else: 
                    if full_quad_coverage == 1:
                        E_tot = E
                    elif full_quad_coverage == 0:
                        E_tot = 0
                
                l = l + 1
                
                # The time is the actual duration of the kick/drop itself
                if l*dt > dur:
                    recovered = 0
            
            # If currently in the recovery phase
            
            if recovered == 0:
                
                E0[i+1] = E0_max - E_drop[2]*np.exp(-dt*k/tc)
                        
                if cf.passthroughElementContact(x,dqel_theta):
                    E_tot = E - E_drop*np.exp(-dt*k/tc)
            
                elif cf.passthroughElementContact(x,sqel_theta):
                    E_tot = E

                else: 
                    if full_quad_coverage == 1:
                        E_tot = E
                    elif full_quad_coverage == 0:
                        E_tot = 0  
                
                k = k + 1
                
                ## Uncomment if a 'recovered' phase is desired, after the
                ## E-field has recovered to within a certain fraction of full                
#
#               rec_fraction = 0.99999
#                               
#                if drop > 0:
#                
#                    if E0[i+1]/E0_max > rec_fraction:
#                        print('Recovered')
#                        recovered = 1
#                        r_i = i
#                        ymaxc_i = ymaxc
#                        yminc_i = yminc
#                        
#                if drop < 0:
#                
#                    if E0_max/E0[i+1] > rec_fraction:
#                        print('Recovered')
#                        recovered = 1
#                        r_i = i
#                        ymaxc_i = ymaxc
#                        yminc_i = yminc
                
        # New force vector
        F = cf.forceDueFields(v,B,E_tot,q)
        
        # Just to make sure nothing goes faster than the speed of light
        
        if np.sqrt(v[0]**2 + v[1]**2) > c:
            print('illegal!!')

        # Marks locations when yprime is within a set distance from 0        
        
        if np.abs(yprime) < yprime_limit and x[2] > 0:
            
            y_max[ymaxc] = x[2]
            ymaxc = ymaxc + 1
        
        if np.abs(yprime) < yprime_limit and x[2] < 0:
            
            y_min[yminc] = x[2]
            yminc = yminc + 1
        
        i = i + 1
        
    print(E0[i]/E0_max)                 # For reference only
    
    # If recovered == 1 was never reached
    
    if r_i == 0:
        r_i = i
        
        # Only used if recovery == 1 is used
        ymaxc_i = ymaxc
        yminc_i = yminc
    
    # To prevent warnings from being throws about r_i not explicitely being int
    r_i = int(r_i)
        
    # Remove zeros
    y_max = y_max[0:ymaxc:1]
    y_min = y_min[0:yminc:1]
        
#==============================================================================
#   Plotting
#==============================================================================

    if make_plots == 1:
    
        ''' Setting useful variables '''
        
        if drop > 0:
            typeAct = "Drop"
        elif drop < 0:
            typeAct = "Kick"
        else:
            typeAct = "Constant"
        
        # Counter used for setting multiple figures
        n = 0
        
        # Convert to mm
        
        y_min = y_min*1000
        y_max = y_max*1000
        
        # Used in adding a text box to the plots
#        props = dict(boxstyle='square', facecolor='wheat', alpha=0.8)
        
        ''' Plotting '''
        
        # Figure
        plt.figure(n)
        n = n + 1
        
        ax = plt.subplot(1,1,1)
        ax.plot(tt*(10**6),E0/1000)
        plt.xlabel('Time ($\mu$s)')
        plt.ylabel('E-field (kV/m)')
        plt.title('E-field Strength - %s: %0.0f Volts'%(typeAct,np.abs(drop)))
        
        plt.ylim(min(E0/1000)-200/1000,max(E0/1000)+200/1000)
        plt.grid()

        if save_plots == 1:
            plt.savefig('../Output/%s/%s_E_Field_%d.png'%(
                        save_folder,plot_save_text,drop),
                        bbox_inches='tight', dpi=300)
            
        plt.figure(n)
        n = n + 1
        
        ax = plt.subplot(1,1,1)
        
        # Used to plot histogram of just the y_maxs after recovered == 1
#        ax.hist(y_max[ymaxc_i:ymaxc:1],50)
        
        # Or plot all y_maxs
        ax.hist(y_max,100)
        plt.title('Histogram of y-max at y\'=0 - %s: %0.0f Volts'%(
                    typeAct,np.abs(drop)))
        plt.xlabel('y-Position (mm)')

        if save_plots == 1:
            plt.savefig('../Output/%s/%s_hist_y_max_%d.png'%(
                        save_folder,plot_save_text,drop),
                        bbox_inches='tight', dpi=300)
            
        plt.figure(n)
        n = n + 1
        
        ax = plt.subplot(1,1,1)
        
        # Used to plot histogram of just the y_mins after recovered == 1
#        ax.hist(y_min[yminc_i:yminc:1],50) 
        
        # Or plot all y_mins
        ax.hist(y_min,100)
        plt.title('Histogram of y-min at y\'=0 - %s: %0.0f Volts'%(
                    typeAct,np.abs(drop)))
        plt.xlabel('y-Position (mm)')

        if save_plots == 1:
            plt.savefig('../Output/%s/%s_hist_y_min_%d.png'%(
                        save_folder,plot_save_text,drop),
                        bbox_inches='tight', dpi=300)
        
        '''
        Plots previously used but no longer as some variables are no longer
        stored permanently, such as a history of the net force on the muon. To
        use some of these, single variables in the code will need to be changed
        to arrays.
        '''

#        # Figure
#        plt.figure(n)
#        n = n + 1
#        
#        ax = plt.subplot(1,1,1)
#        ax.plot(x[:,0],x[:,1], label='Particle Track',lw=lw)
#        plt.xlabel('x-position (m)')
#        plt.ylabel('y-position (m)')  
#        plt.title('Muon Position')
#        
#        # Add single-quad electrodes      
#        
#        count = len(sqel_theta)
#        k = 0
#        M = 100
#        
#        # Plot those for y > 0
#        
#        while k < count/2:
#            
#            xt = np.linspace(
#                sqel_rad[0]*np.cos(sqel_theta[k,0]),
#                sqel_rad[0]*np.cos(sqel_theta[k,1]),
#                M
#            )
#            
#            ax.plot(xt,np.sqrt(sqel_rad[0]**2 - xt**2),'k')
#            
#            xt = np.linspace(
#                sqel_rad[1]*np.cos(sqel_theta[k,0]),
#                sqel_rad[1]*np.cos(sqel_theta[k,1]),
#                M
#            )
#            ax.plot(xt,np.sqrt(sqel_rad[1]**2 - xt**2),'k')
#            
#            k = k + 1
#            
#        # Plot those for y < 0
#            
#        while  k < count:
#            
#            xt = np.linspace(
#                sqel_rad[0]*np.cos(sqel_theta[k,0]),
#                sqel_rad[0]*np.cos(sqel_theta[k,1]),
#                M
#            )
#            
#            ax.plot(xt,-np.sqrt(sqel_rad[0]**2 - xt**2),'k')
#            
#            xt = np.linspace(
#                sqel_rad[1]*np.cos(sqel_theta[k,0]),
#                sqel_rad[1]*np.cos(sqel_theta[k,1]),
#                M
#            )
#            ax.plot(xt,-np.sqrt(sqel_rad[1]**2 - xt**2),'k')
#            
#            k = k + 1
#        
#        # Add double-quad electrodes      
#        
#        count = len(dqel_theta)
#        k = 0
#        M = 100
#        
#        # Plot those for y > 0
#        
#        while k < count/2:
#            
#            xt = np.linspace(
#                dqel_rad[0]*np.cos(dqel_theta[k,0]),
#                dqel_rad[0]*np.cos(dqel_theta[k,1]),
#                M
#            )            
#            ax.plot(xt,np.sqrt(float(dqel_rad[0])**2 - xt**2),'k')
#            
#            xt = np.linspace(
#                dqel_rad[1]*np.cos(dqel_theta[k,0]),
#                dqel_rad[1]*np.cos(dqel_theta[k,1]),
#                M
#            )
#            ax.plot(xt,np.sqrt(dqel_rad[1]**2 - xt**2),'k')
#            
#            k = k + 1
#            
#        # Plot those for y < 0
#            
#        while  k < count:
#            
#            xt = np.linspace(
#                dqel_rad[0]*np.cos(dqel_theta[k,0]),
#                dqel_rad[0]*np.cos(dqel_theta[k,1]),
#                M
#            )            
#            ax.plot(xt,-np.sqrt(dqel_rad[0]**2 - xt**2),'k')
#            
#            xt = np.linspace(
#                dqel_rad[1]*np.cos(dqel_theta[k,0]),
#                dqel_rad[1]*np.cos(dqel_theta[k,1]),
#                M
#            )
#            ax.plot(xt,-np.sqrt(dqel_rad[1]**2 - xt**2),'k')
#            
#            k = k + 1
#        
#        plt.axis('equal') # Prevents a skewed look
#        
#        # Figure
#        plt.figure(n)
#        n = n + 1
#        
#        ax = plt.subplot(1,1,1)
#        ax.plot(x[:,2][d_i:r_i:1]*1000,yprime[d_i:r_i:1]*1000,'r',
#                lw=lw)
#        ax.plot(x[:,2][r_i:steps:1]*1000,yprime[r_i:steps:1]*1000,'g',lw=lw*5)
#        ax.plot(x[:,2][0:d_i:1]*1000,yprime[0:d_i:1]*1000,'b',lw=lw)
#        plt.xlabel('y (mm)')
#        plt.ylabel('y\' (m/m * 1000)')
#        plt.title('y Phase-Space - %s: %0.0f Volts'%(typeAct,np.abs(drop)))
#        plt.grid()
#
#        if save_plots == 1:
#            plt.savefig('Output/%s/%s_y_yprime_%0.2f.png'%(
#                        save_folder,plot_save_text,drop),
#                        bbox_inches='tight', dpi=300)
#        
#        # Figure
#        plt.figure(n)
#        n = n + 1
#        
#        ax = plt.subplot(1,1,1)
#        ax.plot(x[:,2][d_i:r_i:1]*1000,yprime[d_i:r_i:1]*1000,'r',
#                lw=lw)
#        ax.plot(x[:,2][r_i:steps:1]*1000,yprime[r_i:steps:1]*1000,'g',lw=lw*5)
#        ax.plot(x[:,2][0:d_i:1]*1000,yprime[0:d_i:1]*1000,'b',lw=lw)
#        plt.xlabel('y (mm)')
#        plt.ylabel('y\' (m/m * 1000)')
#        plt.title('y Phase-Space - %s: %0.0f Volts'%(typeAct,np.abs(drop)))
#        
#        plt.xlim(37,41)
#        plt.ylim(-1,1)
#        plt.grid()
#
#        # Save the plot(s) if save_plots == 1
#
#        if save_plots == 1:
#            plt.savefig('Output/%s/%s_y_yprime_Q1_%0.2f_zoom.png'%(
#                        save_folder,plot_save_text,drop),
#                        bbox_inches='tight', dpi=300)
#        
#        # Figure
#        plt.figure(n)
#        n = n + 1
#        
#        ax = plt.subplot(1,1,1)
#        ax.plot(x[:,2][d_i:r_i:1]*1000,yprime[d_i:r_i:1]*1000,'r',
#                lw=lw)
#        ax.plot(x[:,2][r_i:steps:1]*1000,yprime[r_i:steps:1]*1000,'g',lw=lw*5)
#        ax.plot(x[:,2][0:d_i:1]*1000,yprime[0:d_i:1]*1000,'b',lw=lw)
#        plt.xlabel('y (mm)')
#        plt.ylabel('y\' (m/m * 1000)')
#        plt.title('y Phase-Space - %s: %0.0f Volts'%(typeAct,np.abs(drop)))
#        
#        plt.xlim(-10,10)
#        plt.ylim(1.1,1.4)
#        plt.grid()
#
#        # Save the plot(s) if save_plots == 1
#
#        if save_plots == 1:
#            plt.savefig('Output/%s/%s_y_yprime_Q2_%0.2f_zoom.png'%(
#                        save_folder,plot_save_text,drop),
#                        bbox_inches='tight', dpi=300)
#        
#        # Figure
#        plt.figure(n)
#        n = n + 1
#        
#        ax = plt.subplot(1,1,1)
#        ax.plot(x[:,2][d_i:r_i:1]*1000,yprime[d_i:r_i:1]*1000,'r',
#                lw=lw)
#        ax.plot(x[:,2][r_i:steps:1]*1000,yprime[r_i:steps:1]*1000,'g',lw=lw*5)
#        ax.plot(x[:,2][0:d_i:1]*1000,yprime[0:d_i:1]*1000,'b',lw=lw)
#        plt.xlabel('y (mm)')
#        plt.ylabel('y\' (m/m * 1000)')
#        plt.title('y Phase-Space - %s: %0.0f Volts'%(typeAct,np.abs(drop)))
#        
#        plt.xlim(-41,-37)
#        plt.ylim(-1,1)
#        plt.grid()
#
#        # Save the plot(s) if save_plots == 1
#
#        if save_plots == 1:
#            plt.savefig('Output/%s/%s_y_yprime_Q3_%0.2f_zoom.png'%(
#                        save_folder,plot_save_text,drop),
#                        bbox_inches='tight', dpi=300)
#        
##        # Figure
#        plt.figure(n)
#        n = n + 1
#        
#        ax = plt.subplot(1,1,1)
#        ax.plot(x[:,2][d_i:r_i:1]*1000,yprime[d_i:r_i:1]*1000,'r',
#                lw=lw)
#        ax.plot(x[:,2][r_i:steps:1]*1000,yprime[r_i:steps:1]*1000,'g',lw=lw*5)
#        ax.plot(x[:,2][0:d_i:1]*1000,yprime[0:d_i:1]*1000,'b',lw=lw)
#        plt.xlabel('y (mm)')
#        plt.ylabel('y\' (m/m * 1000)')
#        plt.title('y Phase-Space - %s: %0.0f Volts'%(typeAct,np.abs(drop)))
#        
#        plt.xlim(-10,10)
#        plt.ylim(-1.4,-1.1)
#        plt.grid()
#
#        # Save the plot(s) if save_plots == 1
#
#        if save_plots == 1:
#            plt.savefig('Output/%s/%s_y_yprime_Q4_%0.2f_zoom.png'%(
#                        save_folder,plot_save_text,drop),
#                        bbox_inches='tight', dpi=300)
#        
#        # Figure
#        plt.figure(n)
#        n = n + 1
#        
#        ax = plt.subplot(1,1,1)
#        ax.plot(s,x[:,2]*1000, label='Particle Track')
#        plt.xlabel('position (m)')
#        plt.ylabel('z-position (mm)')
#        plt.title('Muon Position')
#        
#        # Figure
#        plt.figure(n)
#        n = n + 1
#        
#        ax = plt.subplot(1,1,1)
#        ax.plot(x[:,2][d_i:r_i:1]*1000,E_tot[:,2][d_i:r_i:1]/1000,'r',lw=lw)
#        ax.plot(x[:,2][0:d_i:1]*1000,E_tot[:,2][0:d_i:1]/1000,'b',lw=lw)
#        plt.xlabel('y-Position (mm)')
#        plt.ylabel('E-field (kV/m)')
#        plt.title('E-field Strength - %s: %0.0f Volts'%(typeAct,np.abs(drop)))
#        plt.xlim(-1,1)
#        plt.ylim(-10,10)
#        plt.grid()
#
#        if save_plots == 1:
#            plt.savefig('Output/%s/%s_E_Field_%0.2f.png'%(
#                        save_folder,plot_save_text,drop),
#                        bbox_inches='tight', dpi=300)
                        
#        if make_plots1 == 1:
#            plt.show()
    
    print('Maximum y: %0.05f mm'%(x_max[1]*1000))
    print('Minimum y: %0.05f mm'%(x_max[0]*1000))
    
if __name__ == '__main__':
    
    main()