/*
 ============================================================================
 Name        : CTSim-noise.c
 Author      : German Mato
 Description : Puts gaussian noise into CTSim projection file (*.pj) 
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

float gasdev(long *idum);
float ran1(long *idum);

int main(int argc, char *argv[]) 
{
	FILE *file, *file2;
	int i, j, numDets, numViews, geometry, numDet;
	double focalLength, sourceDetectorLength, angle, calcTime;
	double rotStart, rotInc, detStart, detInc, viewDiameter, viewAngle;	
	unsigned short int headerSize, signature, year, month, day;
	unsigned short int hour, minute, second, remarkSize;
	unsigned short int validSignature = 'P'*256 + 'J';
	char *remark;
	char defaultOutput[]="projections-noise.pj";
	char *output;
	float *data;
	long seed=-1;
	float sigma;
	
	printf("Noise into Projection file.\n");
	printf("G. Mato.\n");

	printf("Noise level=\n");
	scanf("%f",&sigma);
	
	if((argc!=2) && (argc!=3))
	{
		printf("Usage: %s <CTSim projection file> [CTSim projection file]\n",argv[0]);
		printf("Example: %s projections.pj projections-noise.pj", argv[0]);
		return 0;
	}
	
	file = fopen(argv[1], "rb");
	if(file==NULL)
	{
		printf("Unable to open projection file: %s\n",argv[1]);
		return 0;
	}
	
	fread(&headerSize,	2, 1, file);
	fread(&signature,	2, 1, file);
	if(signature!=validSignature)
	{
		printf("Invalid projection file: %s\n", argv[1]);
		return 0;
	}
	
	if(argc==2)
		output = defaultOutput;
	else
		output = argv[2];
	
	file2 = fopen(output,"wb");
	
	if(file==NULL)
	{
		printf("Unable to open output file: %s",output);
		return 0;
	}
	
	fread(&numViews, 			4, 1, file);
	fread(&numDets, 			4, 1, file);
	fread(&geometry, 			4, 1, file);
	fread(&calcTime, 			8, 1, file);
	fread(&rotStart, 			8, 1, file);
	fread(&rotInc, 				8, 1, file);
	fread(&detStart, 			8, 1, file);
	fread(&detInc, 				8, 1, file);
	fread(&viewDiameter, 			8, 1, file);
	fread(&focalLength,			8, 1, file);
	fread(&sourceDetectorLength,		8, 1, file);
	fread(&angle, 				8, 1, file);
	fread(&year,				2, 1, file);
	fread(&month, 				2, 1, file);
	fread(&day,				2, 1, file);
	fread(&hour, 				2, 1, file);
	fread(&minute,				2, 1, file);
	fread(&second, 				2, 1, file);
	fread(&remarkSize, 			2, 1, file);
	
	remark = (char *)malloc(remarkSize);
	fread(remark,1,remarkSize,file);
	fseek(file, headerSize, SEEK_SET);
	
	fwrite(&headerSize,			2, 1, file2);
	fwrite(&signature, 			2, 1, file2);
	fwrite(&numViews, 			4, 1, file2);
	fwrite(&numDets, 			4, 1, file2);
	fwrite(&geometry, 			4, 1, file2);
	fwrite(&calcTime, 			8, 1, file2);
	fwrite(&rotStart, 			8, 1, file2);
	fwrite(&rotInc, 			8, 1, file2);
	fwrite(&detStart, 			8, 1, file2);
	fwrite(&detInc, 			8, 1, file2);
	fwrite(&viewDiameter, 			8, 1, file2);
	fwrite(&focalLength,			8, 1, file2);
	fwrite(&sourceDetectorLength,		8, 1, file2);
	fwrite(&angle, 				8, 1, file2);
	fwrite(&year,				2, 1, file2);
	fwrite(&month, 				2, 1, file2);
	fwrite(&day,				2, 1, file2);
	fwrite(&hour, 				2, 1, file2);
	fwrite(&minute,				2, 1, file2);
	fwrite(&second, 			2, 1, file2);
	fwrite(&remarkSize, 			2, 1, file2);
	
	fwrite(remark,1,remarkSize,file2);
	fseek(file2, headerSize, SEEK_SET);
	
	data = (float *)malloc(sizeof(float)*numDets*numViews);
	for (i=0; i<numViews; i++)
	{
		fread(&viewAngle, 8, 1, file);
		fread(&numDet,	4, 1, file);
		fread(&(data[i*numDets]), 4, numDets, file);

/* aca se incluye el ruido */		
		for(j=0;j<numDets;j++) {data[i*numDets+j]=data[i*numDets+j]+sigma*gasdev(&seed);
			if(j==150) data[i*numDets+j]=0;
                }	
		fwrite(&viewAngle, 8, 1, file2);
		fwrite(&numDet,	4, 1, file2);
		fwrite(&(data[i*numDets]), 4, numDets, file2);
	}
	
	fclose(file);
	fclose(file2);
			
	free(data);
	free(remark);
	return 0;
}

#include <math.h>

float gasdev(long *idum)
{
	float ran1(long *idum);
	static int iset=0;
	static float gset;
	float fac,rsq,v1,v2;

	if (*idum < 0) iset=0;
	if  (iset == 0) {
		do {
			v1=2.0*ran1(idum)-1.0;
			v2=2.0*ran1(idum)-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

float ran1(long *idum)
{
	int j;
	long k;
	static long iy=0;
	static long iv[NTAB];
	float temp;

	if (*idum <= 0 || !iy) {
		if (-(*idum) < 1) *idum=1;
		else *idum = -(*idum);
		for (j=NTAB+7;j>=0;j--) {
			k=(*idum)/IQ;
			*idum=IA*(*idum-k*IQ)-IR*k;
			if (*idum < 0) *idum += IM;
			if (j < NTAB) iv[j] = *idum;
		}
		iy=iv[0];
	}
	k=(*idum)/IQ;
	*idum=IA*(*idum-k*IQ)-IR*k;
	if (*idum < 0) *idum += IM;
	j=iy/NDIV;
	iy=iv[j];
	iv[j] = *idum;
	if ((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
