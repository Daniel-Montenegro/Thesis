#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "allvars.h"

/* 
En este archivo se hacen la lectura de datos. 
Que en este caso es para leer los autovalores de la matriz de marea.
 */

int main(int argc, char *argv[])
{
  
  FILE *in;
  //variables que tiene el archivo de datos
  int dumb, i;
  char line[30];
  int n_x, n_y, n_z;
  int n_nodes;
  long long n_total;
  float dx, dy, dz, x_0, y_0, z_0, **binary;
  char filename[500];
  char *infile;
  
  
  infile = argv[1];
  //Loading file
  if(!(in=fopen(infile, "r")))
    {
      fprintf(stderr, "Problem opening file %s\n", infile);
      exit(1);
    }
  
  printf( "  * The binary file '%s' has been loaded!\n", infile );
  fread(&dumb,sizeof(int),1,in);
  printf("dumb= %d\n",dumb);
  fread(line,sizeof(char)*30,1,in);
  fread(&dumb,sizeof(int),1,in);
  printf("dumb= %d\n",dumb);
  fread(&dumb,sizeof(int),1,in);
  printf("dumb= %d\n",dumb);
  fread(&n_x,sizeof(int),1,in);    
  fread(&n_y,sizeof(int),1,in);    
  fread(&n_z,sizeof(int),1,in);    
  fread(&n_nodes,sizeof(int),1,in);    
  fread(&x_0,sizeof(float),1,in);    
  fread(&y_0,sizeof(float),1,in);    
  fread(&z_0,sizeof(float),1,in);    
  fread(&dx,sizeof(float),1,in);    
  fread(&dy,sizeof(float),1,in);    
  fread(&dz,sizeof(float),1,in);    
  fread(&dumb,sizeof(int),1,in);
  printf("dumb= %d\n",dumb);
  n_total = n_x * n_y * n_z;
  printf("n_x= %d, n_y= %d n_z= %d \n", n_x, n_y, n_z);
  
  if(!(binary=(float **)malloc(n_nodes * sizeof(float*))))
    {
      fprintf(stderr, "problem with array allocation\n");
      exit(1);
    }
  
  fread(&dumb,sizeof(int),1,in);
  fread(&(binary[0]),sizeof(float), n_total, in);
  fread(&dumb,sizeof(int),1,in);
  
  fclose(in);

  sprintf(filename,"%s.ascii",argv[1]);
 
//  for(j=0; j<n_y; j++)
//    for(k=0; k<n_z; k++)
  
  in = fopen(filename,"w");
  
  /*
  for(i=0; i<n_total; i=i+3)
    {
      fprintf(in,"%16.8f %16.8f %16.8f\n",binary[i], binary[i+1], binary[i+2]);
    }
  fclose(in);
  */

  
  //escritura en ascii
  int k,j,n;
  
  for(i=0;i<n_total;i++)
    binary[i]=(float *)malloc(n_nodes * sizeof(float ));

  for(i=0;i<n_x;i++)
    {
      for(j=0;j<n_x;j++)
	{
	  k=0.0
	  n=k+n_x*(j+n_x*i); //n_x=256
	  fprintf(in,"%16.8f %16.8f \n",binary[n], binary[n+1] );
	  
	  //fprintf(in,"\n");
	}
    }
  fclose(in);
  
  
  return 0;
  
}


/*

int k,j;
  
  for(i=0;i<n_total;i++)
    binary[i]=(float *)malloc(n_nodes * sizeof(float ));
  for(i=0;i<n_total;i++)
    {
      for(j=0;j<n_total;j++)
	{
	  for(k=0;k<n_total;k++)
	    fprintf(in,"%16.8f \n",binary[i][j][k]);
	}
    }
  
 */
