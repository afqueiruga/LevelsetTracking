#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef real
#define real double
#endif


void dphidt(real * p, real * dp, int ny, int nx, int lda,
	    real A, real eps) {
  #define RX (lda)
  #define PMN(m,n) ( (m)*RX+(n) )
  #define H0 (1.0)
  #define H1 (1.0)
  #define NP0 (ny)
  #define NP1 (nx)
  for(int m=1; m<NP0-1; m++) {
    for(int n=1; n<NP1-1; n++) {
      real Dc0 = (p[PMN(m-1,n)] - p[PMN(m+1,n)])/(2.0*H0);
      real Dc1 = (p[PMN(m,n-1)] - p[PMN(m,n+1)])/(2.0*H1);
      real gpmag = sqrt(Dc0*Dc0+Dc1*Dc1);
      real n0 = -Dc1/gpmag;
      real n1 = Dc0/gpmag;


      real DDc0 = (p[PMN(m-1,n)] -2.0*p[PMN(m,n)] + p[PMN(m+1,n)])/(H0*H0);
      real DDc1 = (p[PMN(m,n-1)] -2.0*p[PMN(m,n)] + p[PMN(m,n+1)])/(H1*H1);
      real Dc0Dc1 = (p[PMN(m-1,n-1)]-p[PMN(m+1,n-1)]+p[PMN(m+1,n+1)]-p[PMN(m-1,n+1)]
		     )/(4.0*H0*H1);
      real k = ( DDc0*Dc1*Dc1 - 2.0*Dc0*Dc1*Dc0Dc1 + DDc1*Dc0*Dc0 ) 
	   / ( Dc0*Dc0+Dc1*Dc1 + 1e-7);
      dp[PMN(m,n)] = A*gpmag + eps*k;

    }
  }
}

void curvature_filter(real * p, real * dp, int ny, int nx, int lda 
		      int npass, real A, real eps) {
  real DT = 0.02;
  for(int t=0;t<npass;t++) {
    dphidt((real*)imgarray,(real*)didt,A,eps);
    for(int y=1; y<ny-1;y++) {
      for(int x=1;x<nx-1;x++) {
	/* printf("%e\n",didt[y][x]); */
	imgarray[y][x] += DT*didt[y][x];
      }
    }
  }
}
