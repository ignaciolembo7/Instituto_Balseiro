/* Programa que lee una imagen en formato .pgm (ASCII) y  escribe otra ya procesada */

#include <stdlib.h>
#include <stdio.h>
#define Maxline 1000

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
fprintf(fp,"# Creado por programa xxx\n");


fprintf(fp,"%d\t",width_out);
fprintf(fp,"%d\n",height_out);
fprintf(fp,"%d\n",bit_depth_out);

printf("Parametros de Salida: %d \t %d \t %d \n",width_out,height_out,bit_depth_out);

for(i=0;i<width_out*height_out;i++) fprintf(fp,"%d\n",image_pointer_out[i]);

fclose(fp);
}


/* Funcion de Procesamiento */
void process_pgm_file()
{
  int i;

   width_out=width;
   height_out=height;
   bit_depth_out=bit_depth;
   
   image_pointer_out=(int *)malloc(sizeof(int)*width_out*height_out);
  

   // Hace algo para calcular image_pointer_out  
  for(i=0;i<width_out*height_out;i++) image_pointer_out[i]=image_pointer[i];
  
}



/* Funcion de interpolacion bi-lineal. Las coordenadas deben estar entre 0 y 1 */
float inter_lin(float x,float y)
{
int ix,iy,i0;
float dx,dy,a0,a1,a2,a3;
  
if(x <0 || x>=1 || y <0 || y>=1) return(0);

ix= x*width;
iy= y*height;

dx=x*width-ix;
dy=y*height-iy;

i0=ix+iy*width;

a0=image_pointer[i0];
a1=image_pointer[i0+1];
a2=image_pointer[i0+width];
a3=image_pointer[i0+width+1];

return(a0+(a1-a0)*dx+(a2-a0)*dy+(a0+a3-a1-a2)*dx*dy);
}