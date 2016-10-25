# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:25:02 2016

@author: Eric Schmidt
"""

import numpy as np
import matplotlib.pyplot as plt
import callable_functions as cf

def main():
    
    make_plots = 1                      # 1 to make plots, 0 to skip
    save_plots = 1                      # 1 to save plots, 0 to skip
#    save_output = 1                     # 1 to save data to csv
    vary = 1                            # 1 to include the change in E-field
    plot_save_text = "real_quad_entire" # Text to distinguish specific saves
    full_quad_coverage = 0              # 1 to place quads around entire ring

    yprime_limit = 0.00001              # To record y_max
    lw = 0.1                            # Thickness of plotted lines
    steps = 5*10**6                     # Number of steps for integration
    dt = 10**-11                        # Timestep for integration
    d_i = int(steps/5)
    tt = np.linspace(0,dt*steps,steps)  # For plotting
    tt = np.reshape(tt,(steps,1))
    p_mag = 3.09435*10**9               # (eV/c) Possible positron momentums
    q = 1                               # (e) Positron charge
    c = 2.99792458*10**8                # (m/s) Speed of light
    m = 105.65837*10**6                 # (eV/c**2) Muon mass
    B = np.array([0,0,1.4513])          # (T) Magnetic field
    R = 7.112                           # (m) Radius of ideal orbit
#    voltage = 30000                     # (V) Voltage
    theta = 0                           # (rads) Initial theta in global coords
    n = .142                            # () Used in E-field
    recovered = -2                      # 0 if E-field not recovered from drop
    if vary == 1:
        drop = 180                      # Drop in voltage
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
    
    ''' Variables for both quads '''
    
    qel_depth = 0.5*10**-3      # (m) Electrode thickness
    qel_rad_start = 50*10**-3   # (m) Starting distance in from R
    
    ''' Single-quad electrodes (without edge curls) '''
    
    sqel_rad = np.array([R - (qel_rad_start + qel_depth),R - qel_rad_start])
    sqel_theta_base = 90 - np.array([31.89,44.89])-4
    sqel_theta = 90 - np.array([31.89,44.89])-4
    
    i = 1
    while i < 4:
        sqel_theta = np.row_stack((
            sqel_theta, sqel_theta_base + i*90
        ))
        i = i + 1
        
    sqel_theta = sqel_theta*np.pi/180
    
    ''' Double-quad electrodes (without edge curls) '''
    
    dqel_rad = np.array([R - (qel_rad_start + qel_depth),R - qel_rad_start])
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
    v = beta*c                       # (m/s) Initial velcoty
    s = dt * np.sqrt(v[0]**2 + v[1]**2)
    
    # Magnitude of relativistic beta vector
    
#    beta_mag = np.zeros(1)                  # ()
    beta_mag = cf.mag(beta)
    
    # Relativistic gamma
    
    gamma = cf.beta2Gamma(beta_mag)
    
    # Electric field
    
#    E = np.zeros((steps,3))                 # (V/m) Initialize electric field
    E = cf.getElectricField(x,R,B,n)  # Set initial electric field values
    
    E_tot = cf.getElectricField(x,R,B,n)  # Set initial electric field values
    
    E0 = np.zeros((steps))
    E0[0] = cf.getElectricField(x,R,B,n)[2] # Set initial electric field
    
    E0_max = cf.getElectricField(x,R,B,n)[2] # Set initial electric field
    
    # Force vector
    
#    F = np.zeros((steps,3))                 # (N) Initialze force array
    F = cf.forceDueFields(v,B,E,q) # Set initial force due fields    
    
    E_drop = np.array([0,0,drop])/0.1
    drop_slope = E_drop/(1.56*10**-7)        # Electric field drop rate
    
    y_max = np.zeros((int(steps),1))
    ymaxc = 0                                    # y_max counter
    y_min = np.zeros((int(steps),1))
    yminc = 0                                    # y_max counter
    
    x_max = np.zeros((2))                       # x[0] = -y, x[1] = +y
    
    # Loop counter
    i = 0
    
    # Counter for E-field exponential recovery
    k = 0
    
    # Counter for E-field drop
    l = 0
    
    ''' RK4 work '''
    
    while i < steps - 1:
        
        # Relativistic mass to find the acceleration from the force
        rmass = m*gamma/c**2   
        
        a = F/(rmass)
        dv1 = dt * a
        
        a = cf.forceDueFields(v + dv1/2,B,E_tot,q)/(rmass)
        dv2 = dt * a
        
        a = cf.forceDueFields(v + dv2/2,B,E_tot,q)/(rmass)
        dv3 = dt * a
        
        a = cf.forceDueFields(v + dv3,B,E_tot,q)/(rmass)
        dv4 = dt * a
        
        # New velocity vector
        
        v_old = v
        v = v_old + (dv1 + 2*dv2 + 2*dv3 + dv4) / 6
        
        x_old = x
        
        # New position vector
        x = x_old + dt * v
        
        if x[2] < 0 and x[2] < x_max[0]:
            x_max[0] = x[2]
        if x[2] > 0 and x[2] > x_max[1]:
            x_max[1] = x[2]
        
        gamma = cf.beta2Gamma(cf.mag(v)/c)
        
        # Get the electric field based on position
        E = cf.getElectricField(x,R,B,n)
        
        if recovered == -2 or recovered == 1:
            
            E0[i+1] = E0_max
                        
            if cf.passthroughElementContact(x,dqel_theta):
                E_tot = E
        
            elif cf.passthroughElementContact(x,sqel_theta):
                E_tot = E

            else: 
                if full_quad_coverage == 1:
                    E_tot = E
                elif full_quad_coverage == 0:
                    E_tot = 0
            
        if vary == 1:
        
            if i > d_i and recovered == -2:
                recovered = -1
                
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
                
                if l*dt > 1.56*10**-7:
                    recovered = 0
            
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
                
#                if drop > 0:
#                
#                    if E0[i+1]/E0_max > 0.99999:
#                        print('Recovered to within 0.00001%')
#                        recovered = 1
#                        r_i = i
#                        ymaxc_i = ymaxc
#                        yminc_i = yminc
#                        
#                if drop < 0:
#                
#                    if E0_max/E0[i+1] > 0.99999:
#                        print('Recovered to within 0.00001%')
#                        recovered = 1
#                        r_i = i
#                        ymaxc_i = ymaxc
#                        yminc_i = yminc
                
        # New force vector
        F = cf.forceDueFields(v,B,E_tot,q)
        
        # New arc position
        
        s_old = s
        s = s_old + dt * np.sqrt(v[0]**2 + v[1]**2)
        
        if np.sqrt(v[0]**2 + v[1]**2) > c:
            print('illegal!!')
        
        yprime = (x[2] - x_old[2])/(s - s_old)
        
        if np.abs(yprime) < yprime_limit and x[2] > 0:
            
            y_max[ymaxc] = x[2]
            ymaxc = ymaxc + 1
        
        if np.abs(yprime) < yprime_limit and x[2] < 0:
            
            y_min[yminc] = x[2]
            yminc = yminc + 1
        
        i = i + 1
        
    print(E0[i]/E0_max)
    
    if r_i == 0:
        r_i = i
        ymaxc_i = ymaxc
        yminc_i = yminc
        
    r_i = int(r_i)
        
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
            plt.savefig('Output/%s/%s_E_Field_%d.png'%(
                        save_folder,plot_save_text,drop),
                        bbox_inches='tight', dpi=300)
            
        plt.figure(n)
        n = n + 1
        
        ax = plt.subplot(1,1,1)
#        ax.hist(y_max[ymaxc_i:ymaxc:1],50)
        ax.hist(y_max,100)
        plt.title('Histogram of y-max at y\'=0 - %s: %0.0f Volts'%(
                    typeAct,np.abs(drop)))
        plt.xlabel('y-Position (mm)')
#        plt.xlim(0.040115,0.040125)
#        plt.xlim(0.040116,0.040122)
#        plt.xlim(0.039878,0.039883)
#        plt.xlim(0.03995,0.04004) # Contains everything

        if save_plots == 1:
            plt.savefig('Output/%s/%s_hist_y_max_%d.png'%(
                        save_folder,plot_save_text,drop),
                        bbox_inches='tight', dpi=300)
            
        plt.figure(n)
        n = n + 1
        
        ax = plt.subplot(1,1,1)
#        ax.hist(y_min[yminc_i:yminc:1],50)
        ax.hist(y_min,100)
        plt.title('Histogram of y-min at y\'=0 - %s: %0.0f Volts'%(
                    typeAct,np.abs(drop)))
        plt.xlabel('y-Position (mm)')
#        plt.xlim(-0.040003,-0.039996)
#        plt.xlim(-0.040003,-0.039998)
#        plt.xlim(-0.040002,-0.039988)
#        plt.xlim(-0.04002,-0.03988) # Contains everything

        if save_plots == 1:
            plt.savefig('Output/%s/%s_hist_y_min_%d.png'%(
                        save_folder,plot_save_text,drop),
                        bbox_inches='tight', dpi=300)
        
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
##        # Figure
##        plt.figure(n)
##        n = n + 1
##        
##        ax = plt.subplot(1,1,1)
##        ax.plot(s,x[:,2]*1000, label='Particle Track')
##        plt.xlabel('position (m)')
##        plt.ylabel('z-position (mm)')
##        plt.title('Muon Position')
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