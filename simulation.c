
#include "axi.h"
#include "embed.h"
#include "navier-stokes/centered.h"
#include "tracer.h"

scalar f[];
scalar *tracers = {f};
double Reynolds;
int maxlevel = 9;
face vector muv[];

double D = 1, U0 = 1.;

int main(int argc, char *argv[])
{
  Reynolds = atof(argv[1]);
  L0 = 10;
  init_grid(1 << maxlevel);
  mu = muv;
  DT = 1e-2;
  run();
}

event properties(i++)
{
  foreach_face()
      muv.x[] = fm.x[] * D * U0 / Reynolds;
}

// u.n[left]  = dirichlet(max(0., 2*U0*(1. - sq(y/(D/2.)))));
u.n[left] = neumann(0.);
// u.t[left]  = dirichlet(0.);
u.t[left] = neumann(0.);
double P_in = 1.;
p[left]    = dirichlet(P_in);
pf[left]   = dirichlet(P_in);

u.n[right] = neumann(0.);
u.t[right] = neumann(0.);
p[right]   = dirichlet(0.);
pf[right]  = dirichlet(0.);

u.n[bottom] = dirichlet(0.);
u.t[bottom] = neumann(0.);
p[bottom]   = neumann(0.);
pf[bottom]  = neumann(0.);

// u.n[top] = dirichlet(0.);
// u.t[top] = dirichlet(0.);
// p[top]   = neumann(0.);
// pf[top] = neumann(0.);

u.n[embed] = dirichlet(0.);
u.t[embed] = dirichlet(0.);
p[embed]   = neumann(0.);
pf[embed]  = neumann(0.);

scalar ux_old[], uy_old[];

event init(t = 0)
{


  solid(cs, fs, D / 2. - y);
  boundary({cs, fs});

  foreach ()
  {
    u.x[] = 0.;
    u.y[] = 0.;
    ux_old[] = u.x[];
    uy_old[] = u.y[];
  }
}


event logfile(i++)
    fprintf(stderr, "%d %g %d %d\n", i, t, mgp.i, mgu.i);

double umax_prev = 0.;


event steady_check(i += 50)
{
  double max_dt_change = 0.; // time convergence
  double umax = 0.;          // centerline velocity

  foreach ()
  {
    double du = fabs(u.x[] - ux_old[]) + fabs(u.y[] - uy_old[]);
    if (du > max_dt_change)
    {
      max_dt_change = du;
    }

    ux_old[] = u.x[];
    uy_old[] = u.y[];
    
    if (u.x[] > umax){
      umax = u.x[];
    }
  }

  double d_umax = fabs(umax - umax_prev);

  fprintf(stderr,
          "i=%d t=%g max_dt=%g umax=%g d_umax=%g\n",
          i, t, max_dt_change, umax, d_umax);

  // FINAL CONDITION
  if ((max_dt_change < 1e-4 && d_umax < 1e-4 && i > 200) || t>10)
  {
    fprintf(stderr, "Steady + Fully Developed reached\n");
    return 1;
  }

  umax_prev = umax;
}

event final_output(t = end)
{
  char filename[100];
  sprintf(filename, "steady_state_Re%.0f.csv", Reynolds);
  FILE *fp = fopen(filename, "w");
  fprintf(fp, "x,y,ux,uy,p,cs,f\n");
  foreach ()
    if (cs[] > 0.)
      fprintf(fp, "%g,%g,%g,%g,%g,%g,%g\n",
              x, y, u.x[], u.y[], p[], cs[], f[]);
  fclose(fp);
  fprintf(stderr, "Final output written to steady_state.csv\n");
}

event end(t = 10.) 
{
  fprintf(stderr, "Simulation complete at t=%g\n", t);
}


event adapt(i++)
{
  adapt_wavelet({cs, u, f}, (double[]){1.e-2, 3.e-2, 3.e-2, 3.e-2}, maxlevel, 4);
}



