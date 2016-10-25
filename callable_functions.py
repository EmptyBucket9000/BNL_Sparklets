# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:27:30 2016

@author: Eric Schmidt
"""

import numpy as np

#==============================================================================
# Global Constants
#==============================================================================

c = 2.99792458*10**8 # (m/s) Speed of light

#==============================================================================
# Particle movement and position
#==============================================================================

## Calculate the Lorentz force on the particle

def forceDueFields(v,B,E,q):
    
    F = (q*(E + np.cross(v,B)))
    
    return F
    
## Get the electric field based on position

def getElectricField(x,R,B,n):
    
    E = np.array([0,0,-((n*c*mag(B))/R)*x[2]]) #/0.41 # when quads added in
    
    return E
    
def inQuad(E_tot,E,x,E_drop,dqel_theta,sqel_theta):
                        
    if passthroughElementContact(x,dqel_theta):
        E_tot = E - E_drop
            
    elif passthroughElementContact(x,sqel_theta):
        E_tot = E
        
    else: E_tot = E
    
    return E_tot

## Checks if inside a quad it can pass through
    
def passthroughElementContact(x,el_theta):    

    # Get particle's r and theta positions
    
    theta = getParticleTheta(x)

    # Check if particle is inside an electrode

    for row in el_theta:
        if row[0] > theta and row[1] < theta:
                
            return True
    
## Get the angle of the particle position
    
def getParticleTheta(x):
    
    # Angle of particle position
    theta = np.arctan2(x[1],x[0])
    
    # Translate to 0-2pi
    if theta < 0:
        theta = 2*np.pi + theta
    
    return theta

#==============================================================================
# Misc. functions
#==============================================================================
    
## Return the positron momentum at decay
    
def getPositronMomentumAtDecay(theta,p):

    # Currently assumes x' = y' = z' = 0
    
    # Convert to momentum vector based on position
    p = np.array([p*np.sin(theta),-p*np.cos(theta),0])
    
    return p

## Get momentum from total energy and mass

def energy2Momentum(E,m):
    
    if E >= m:
        p = np.sqrt(E**2 - m**2)
    else:
        p = 0
    
    return p

## Get total energy from velocity and mass

def velocity2Energy(v,m):
    
    v_mag = mag(v)
    beta = v_mag/c
    p = beta2Momentum(beta,m)
    E_tot = momentum2Energy(p,m)
    
    return E_tot

## Get total energy from momentum and mass

def momentum2Energy(p,m):

    p_mag = mag(p)
    E_tot = np.sqrt(p_mag**2 + m**2)
 
    return E_tot

## Convert beta to momentum

def beta2Momentum(beta,m):

    v = beta*c    
    v_mag = mag(v)
    beta_mag = v_mag/c
#    beta_mag = mag(beta)
    gamma = beta2Gamma(beta_mag)
    p = gamma*m*beta
    
    return p

## Convert total energy to relativistic gamma

def energy2Gamma(E,m):
    
    gamma = E/m
    
    return gamma
    
## Convert relativistic beta to relativistic gamma
    
def beta2Gamma(beta):

    gamma = 1/np.sqrt(1-mag(beta)**2)
    
    return gamma
    
## Convert relativistic gamma to relativistic beta

def gamma2Beta(g):

    beta = np.sqrt((g**2-1)/g**2)
    
    return beta
    
## Convert particle momentum to relativistic beta vector
    
def momentumScalar2Beta(p,m):

    beta = np.sqrt(p**2/(m**2 + p**2))
    
    return beta
    
def momentum2Beta(p,m):
    
    p = np.array([p[0],p[1],p[2]])
    p_norm = mag(p)
    
    beta = 1/np.sqrt((m/p_norm)**2 + 1)
    v = beta*c
    v = v*(p/p_norm)
    beta = v/c

    return beta
    
# Get the magnitude of some vector
    
def mag(v):
    
    return np.sqrt(np.dot(v,v))