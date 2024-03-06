/* Programa que calcula la transformada de Radon de una imagen */

#include <stdlib.h>
#include <stdio.h>
#define Maxline 1000
#define PI 3.1415927

void read_pgm_file(char *filename);
void write_pgm_file(char *filename);
void process_pgm_file();
float inter_lin(float,float);

/* Variables globales */
int width, height, bit_depth;
int width_out, height_out, bit_depth_out;
int *image_pointer,*image_pointer_out;



int main(int argc, char *argv[]) 
{
if(argc != 3) {printf("error en los datos de entrada\n"); abort();}
 
read_pgm_file(argv[1]);

process_pgm_file();

write_pgm_file(argv[2]);
 
return 0;
} 


/* Funcion de Lectura */
void read_pgm_file(char *filename) 
{
int i,c,c0;

/* Abre el archivo de lectura */
FILE *fp = fopen(filename, "r");


/* Lee la  linea inicial y los comentarios */
for(i=0;i<Maxline-1 && (c=fgetc(fp)) != '\n'; i++);
while((c0=fgetc(fp))=='#')
{printf("%c",c0); {for(i=0;i<Maxline-1 && (c=fgetc(fp)) != '\n'; i++) printf("%c",c);} printf("\n");}
ungetc(c0,fp);

/* Lee los parametros de la imagen */
fscanf(fp,"%d",&width);
fscanf(fp,"%d",&height);
fscanf(fp,"%d",&bit_depth);

printf("Parametros de Entrada: %d \t %d \t %d \n",width,height,bit_depth);

image_pointer=(int *)malloc(sizeof(int)*width*height);

for(i=0;i<width*height;i++) fscanf(fp,"%d",&image_pointer[i]);

fclose(fp);
}

/* Funcion de Escritura */
void write_pgm_file(char *filename)
{
int i;

/* Abre el archivo de escritura */
FILE *fp = fopen(filename, "w");

/* Escribe  las dos lineas iniciales */
fprintf(fp,"P2\n");
fprintf(fp,"# Creado por programa radon.c\n");


fprintf(fp,"%d\t",width_out);
fprintf(fp,"%d\n",height_out);

float max=0;
for (i=0;i<width_out*height_out;i++) if(image_pointer_out[i]>max) max=image_pointer_out[i];
bit_depth_out=(int)(max+1);
fprintf(fp,"%d\n",bit_depth_out);

printf("Parametros de Salida: %d \t %d \t %d \n",width_out,height_out,bit_depth_out);

for(i=0;i<width_out*height_out;i++) fprintf(fp,"%d\n",image_pointer_out[i]);

fclose(fp);
}

#include<math.h>
/* Funcion de Procesamiento */
void process_pgm_file()
{
  int i,j,k,nsum,i0;
  double theta,r,x,y,s,l,sum,sum_max;

  /* Cantidad de sumandos para el calculo de la integral */
  nsum=250;
  /* Longitud del banco de detectores */
  r=sqrt(2.0);
  
  printf("Numero de angulos=\n");
  scanf("%d",&height_out);
  printf("Numero de detectores=\n");
  scanf("%d",&width_out);
  sum_max=0;
  
  image_pointer_out=(int *)malloc(sizeof(int)*width_out*height_out);
  
  for(i=0;i<height_out;i++)
  {
    theta=i*PI/height_out; 
    for(j=0;j<width_out;j++)
      {
	s=(j*r)/width_out-r/2.;
	sum=0;
	for(k=0;k<nsum;k++)
	{
	  l=(k*r)/nsum-r/2.;
	  
	  x=l*sin(theta)+s*cos(theta);
	  y=-l*cos(theta)+s*sin(theta);
	  
	  sum=sum+inter_lin(x+0.5,y+0.5);
	}
	sum=sum*r/nsum;
	i0=j+i*width_out;
	image_pointer_out[i0]=(int)(sum);
      }
  }
}



/* Funcion de interpolacion bi-lineal. Las coordenadas deben estar entre 0 y 1 */
float inter_lin(float x,float y)
{
int ix,iy,i0;
float dx,dy,a0,a1,a2,a3;
  
if(x <0 || x>=1 || y <0 || y>=1) return(0.0);

ix= x*width;
iy= y*height;

dx=x*width-ix;
dy=y*height-iy;

i0=ix+iy*width;
if(ix<width-1 && iy<height-1)
{
	a0=image_pointer[i0];
	a1=image_pointer[i0+1];
	a2=image_pointer[i0+width];
	a3=image_pointer[i0+width+1];
}
else
{
	a0=image_pointer[i0];
	a1=image_pointer[i0];
	a2=image_pointer[i0];
	a3=image_pointer[i0];
}
return(a0+(a1-a0)*dx+(a2-a0)*dy+(a0+a3-a1-a2)*dx*dy);
}
