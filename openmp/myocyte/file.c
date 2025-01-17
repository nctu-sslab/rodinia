//========================================================================================================================
//	INCLUDE/DEFINE
//========================================================================================================================

// #include <stdlib.h>
// #include <string.h>
#include <stdio.h>
// #include <math.h>
// #include <time.h>
// #include <sys/types.h>
// #include <dirent.h>
// #include <errno.h>

//========================================================================================================================
//	WRITE FUNCTION
//========================================================================================================================

void mywrite(char *filename, fp *input, int data_rows, int data_cols, int major,
           int data_range) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    FILE *fid;
    int i, j;

    //=====================================================================
    //	CREATE/OPEN FILE FOR WRITING
    //=====================================================================

    fid = fopen(filename, "w");
    if (fid == NULL) {
        printf("The file was not created/opened for writing\n");
        return;
    }

    //=====================================================================
    //	WRITE VALUES TO THE FILE
    //=====================================================================

    // if matrix is saved row major in memory (C)
    if (major == 0) {
        for (i = 0; i < data_rows; i++) {
            for (j = 0; j < data_cols; j++) {
                fprintf(fid, "%f ", (fp)input[i * data_cols + j]);
            }
            fprintf(fid, "\n");
        }
    }
    // if matrix is saved column major in memory (MATLAB)
    else {
        for (i = 0; i < data_rows; i++) {
            for (j = 0; j < data_cols; j++) {
                fprintf(fid, "%f ", (fp)input[j * data_rows + i]);
            }
            fprintf(fid, "\n");
        }
    }

    //=====================================================================
    //	CLOSE FILE
    //=====================================================================

    fclose(fid);
}

//========================================================================================================================
//	READ FUNCTION
//========================================================================================================================

void myread(char *filename, fp *input, int data_rows, int data_cols, int major) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    FILE *fid;
    int i, j;
    char c;
    fp temp;

    //=====================================================================
    //	OPEN FILE FOR READING
    //=====================================================================

    fid = fopen(filename, "r");
    if (fid == NULL) {
        printf("The file was not opened for reading\n");
        return;
    }

    //=====================================================================
    //	READ VALUES FROM THE FILE
    //=====================================================================

    if (major == 0) { // if matrix is saved row major in memory (C)
        for (i = 0; i < data_rows; i++) {
            for (j = 0; j < data_cols; j++) {
                fscanf(fid, "%f", &temp);
                input[i * data_cols + j] = (fp)temp;
            }
        }
    } else { // if matrix is saved column major in memory (MATLAB)
        for (i = 0; i < data_rows; i++) {
            for (j = 0; j < data_cols; j++) {
                fscanf(fid, "%f", &temp);
                input[j * data_rows + i] = (fp)temp;
            }
        }
    }

    //=====================================================================
    //	CLOSE FILE
    //=====================================================================

    fclose(fid);
}
