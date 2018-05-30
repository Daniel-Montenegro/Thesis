#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define GB 1024*1024*1024.0
#define X 0
#define Y 1
#define Z 2

//#include "allvars.h"

/* 
En este archivo se hacen la lectura de datos. 
Que en este caso es para leer los autovalores de la matriz de marea.
 */

/***********
COMPILACION:
 gcc -Wall lectura_datos.c -o lectura_datos.X
 ***********/

/*************************
ESTRUCTURA
**************************/
/*
typedef struct
{
  float pos[3];
}Particulas;

Particulas *part;
*/

//RUTINAS//
int escritura_datos(int i,int j, int n_x,float *Eigen);
int asignar_matrix(char *argv[], int n_x);

int main(int argc, char *argv[])
{
  
  FILE *in;
  //variables que tiene el archivo de datos
  int dumb, i,j;
  char line[30];
  int n_x, n_y, n_z;
  int n_nodes, n_total;
  int eigen;
  float dx, dy, dz, x_0, y_0, z_0;
  //float *Eigen;
  float  *Eigen1, *Eigen2, *Eigen3;
  float *enviroment; //-->entorno
  char filename[500];
  char *infile;
  float Eigen_th=0.265; //----> threshold value

  
  /*===========================
    SE HACE UN CICLO PARA LEER LOS TRES ARCHIVOS DE AUTOVALORES
   =============================*/
  for(eigen=0;eigen<3;eigen++)
    {
    //Para que me leea los tres archivos de los autovalores
    sprintf(filename,"%s_%d",argv[1],eigen+1);
    
  //Cargando el archivo
    if(!(in=fopen(filename, "r")))
      {
	fprintf(stderr, "Problem opening file %s\n", filename);
	exit(1);
      }
    //Leyendo la estructura de datos
    printf( "  * The binary file '%s' has been loaded!\n", filename );
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
    printf("dx= %f, dy= %f dz= %f \n", dx, dy, dz);
    printf("x_0= %f, y_0= %f z_0= %f \n", x_0, y_0, z_0);
    printf("n_nodes= %d \n", n_nodes);
    printf("n_total= %d \n", n_total);
    
    //===CUANTO DE LA RAM ESTA CONSUMIENDO LOS DATOS====//
    printf("Allocating %lf Gb for binary pointer\n",1.0*n_x*sizeof(float)/(GB));

    /*******************
     *******************
    ASIGNAR EL VALOR PARA CADA AUTOVALOR
    ********************
     *******************/
    
    //=====EIGENVALOR 1 =========//
    if(eigen==0){
      if(!(Eigen1=(float *) malloc(n_total * sizeof(float))))
	{
	  fprintf(stderr, "problem with array allocation\n");
	  exit(1);
	}
     
      fread(&dumb,sizeof(int),1,in);
      printf("dumb= %d\n",dumb);
      fread(&(Eigen1[0]), sizeof(float), n_total, in);
      fread(&dumb,sizeof(int),1,in);
      printf("dumb= %d\n",dumb);
    }

    //=====EIGENVALOR 2 =========//
    if(eigen==1){
      if(!(Eigen2=(float *) malloc(n_total * sizeof(float))))
	{
	  fprintf(stderr, "problem with array allocation\n");
	  exit(1);
	}
      fread(&dumb,sizeof(int),1,in);
      printf("dumb= %d\n",dumb);
      fread(&(Eigen2[0]), sizeof(float), n_total, in);
      fread(&dumb,sizeof(int),1,in);
      printf("dumb= %d\n",dumb);
    }
    //=====EIGENVALOR 3 =========//
    if(eigen==2){
      if(!(Eigen3=(float *) malloc(n_total * sizeof(float))))
	{
	  fprintf(stderr, "problem with array allocation\n");
	  exit(1);
	}
               
      fread(&dumb,sizeof(int),1,in);
      printf("dumb= %d\n",dumb);
      fread(&(Eigen3[0]), sizeof(float), n_total, in);
      fread(&dumb,sizeof(int),1,in);
      printf("dumb= %d\n",dumb);
    }

    fclose(in);
  } //<----final ciclo for

  /************************************
  *************************************
  COMPARAR LOS VALORES DE LOS AUTOVALORES, Y ASIGNAR EN LA MATRIX
  DE ENVIROMENT. 
  **************************************
  *************************************/

  //Alojar en memoria la matrix de enviroment(entorno)
  if(!(enviroment=(float *) malloc(n_total * sizeof(float))))
    {
      fprintf(stderr, "problem with array allocation\n");
      exit(1);
    }
  
  //Condiciones para asignar valores a cada entorno dependiendo del
  //autovalor de esa posicion.
  for(i=0;i<n_total;i++)
    {
      //========VOID=======//
      if(Eigen3[i]<=Eigen2[i]&& Eigen2[i]<=Eigen1[i] && Eigen1[i]<=Eigen_th)
	{
	  enviroment[i]=0.0; //void
	}
      //=======SHEET======//
      if(Eigen3[i]<=Eigen2[i]&& Eigen2[i]<= Eigen_th && Eigen_th<=Eigen1[i])
	{
	  enviroment[i]=1.0; //sheet
	}
      //=====FILAMENT=====//
      if(Eigen3[i]<Eigen_th && Eigen_th<Eigen2[i] && Eigen2[i]<Eigen1[i])
	{
	  enviroment[i]=2.0; //filament
	}
      //=======KNOT=======//
      if(Eigen_th<Eigen3[i]&& Eigen3[i]<Eigen2[i] && Eigen2[i]<Eigen1[i])
	{
	  enviroment[i]=3.0; //knot
	}
      
    }
  
  /****************************
   GENERAR MATRIX DONDE SE GUARDE LOS EIGENVALORES Y POSICIONES
   ****************************/
   FILE *Enviroment;
  
  Enviroment=fopen("enviroment.dat","w");
  
  //se inicializa contadores para generar una matrix en 2D
  int contmin,contmax;
  
  //  n_x=4;
  printf("voy a entrar al ciclo\n");
  for(i=0;i<n_x;i++)
    {
      for(j=0;j<n_x;j++)
	{
	  //printf("%d \t %d\n",i,j+n_x*i);
	  fprintf(Enviroment,"%f \t %f \t %d \t %d \n",enviroment[i],enviroment[j+n_x*i],i,j);
	  //fprintf(Enviroment,"%f \t %f  \n",enviroment[i],enviroment[i+n_x*j]);
	}
    }
  printf("sali del ciclo\n");
  
  free(Enviroment);
  
  
  //====Asigna valores a la matriz====//
  //====Se concidera solo para Eigen1
  sprintf(filename,"%s.ascii",argv[1]);
  
  FILE *pf; 
  pf = fopen(filename,"w");
  
  //escritura en ascii
  int k,n;
    
  for(i=0;i<n_x;i++)
    {
      for(j=0;j<n_x;j++)
	{
	  k=0; //tomando slices con z=0
	  n=k+n_x*(j+n_x*i); //n_x=256
	  //m=k+n_x*(i+n_x*i)
	  if(n >= n_total)
	    {
	      printf("Vos es que sos marica o que?\n");
	      exit(0);
	    }
	  
	  //fprintf(pf,"%f  %f \n ",part[n].pos[X],part[i].pos[Y] );
	  fprintf(pf,"%f ",Eigen1[n] );
	  
	  
	  //fprintf(in,"\n");
	}
      fprintf(in,"\n");
    }
  printf("el valor de n= %d \n",n);
  fclose(pf); 


  free(Eigen1);
  free(Eigen2);
  free(Eigen3);

  
  return 1;
  printf("holii\n");  
  
}
/*
int escritura_datos(int i,int j, int n_x,float *Eigen1)
{
  
  FILE *finalData;
  
  finalData=fopen("datos_lindos.dat","w");
  
  //se inicializa contadores para generar una matrix en 2D
  int contmin,contmax;
  
  //  n_x=4;
  for(i=0;i<n_x;i++)
    {
      for(j=0;j<n_x;j++)
	{
	  //printf("%d \t %d\n",i,j+n_x*i);
	  fprintf(finalData,"%f \t %f \n",Eigen1[i],Eigen1[i+n_x*j]);
	}
    }
  
  
  free(finalData);
  
  return 0;
  
}

*/

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

======================================

for(i=0;i<n_total;i++)
    {
     fread(&(part[i].pos[0]), sizeof(float), 3, in);
    }
    

  
 */

 
    /*
      for(i=0; i<n_total; i=i+3)
    {
    fprintf(in,"%16.8f %16.8f %16.8f\n",binary[i], binary[i+1], binary[i+2]);
    }
    fclose(in);
  */
