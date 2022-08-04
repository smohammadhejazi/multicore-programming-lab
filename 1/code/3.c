
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define PAD 8 
#define NUM_THREADS 8

const long int VERYBIG = 50000;
// ***********************************************************************
int main(void)
{
    #ifndef _OPENMP
        printf("OpenMP is not supported, sorry!\n");
        getchar();
        return 0;
    #endif
    
	int i, n, mainSum;
	long int j, k;
	double sumx, sumy, mainTotal;
	double starttime, elapsedtime;
	// -----------------------------------------------------------------------
	// Output a start message
	printf("Serial Timings for %d iterations\n\n", VERYBIG);
	// repeat experiment several times
	for (i = 0; i<10; i++)
	{
		// get starting time56 x CHAPTER 3 PARALLEL STUDIO XE FOR THE IMPATIENT
		starttime = omp_get_wtime();
		// reset check sum & running total
        long int sum[NUM_THREADS][PAD] = {0};
		double total[NUM_THREADS][PAD] = {0.0};
		// Work Loop, do some work by looping VERYBIG times
	    omp_set_num_threads(NUM_THREADS);
        #pragma omp parallel for private(k, sumx, sumy)
		for (j = 0; j<VERYBIG; j++)
		{
            int id;
            id = omp_get_thread_num();
			// increment check sum
			sum[id][0] += 1;
			// Calculate first arithmetic series
			sumx = 0.0;
			for (k = 0; k<j; k++)
				sumx = sumx + (double)k;
			// Calculate second arithmetic series
			sumy = 0.0;
			for (k = j; k>0; k--)
				sumy = sumy + (double)k;
			if (sumx > 0.0)total[id][0] = total[id][0] + 1.0 / sqrt(sumx);
			if (sumy > 0.0)total[id][0] = total[id][0] + 1.0 / sqrt(sumy);
		}

        for (n = 0, mainTotal = 0; n<NUM_THREADS; n++) mainTotal += total[n][0];
        for (n = 0, mainSum = 0; n<NUM_THREADS; n++) mainSum += sum[n][0];
		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;
		// report elapsed time
		printf("Time Elapsed: %f Secs, Total = %lf, Check Sum = %ld\n",
			elapsedtime, mainTotal, mainSum);
	}
	// return integer as required by function header
	return 0;
}
