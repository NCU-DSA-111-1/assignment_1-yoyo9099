#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../inc/genann.h"

int summondata() {
    int data;
    data = rand() % 16;
    return data;
}
int binarydata(int data, int shift) {
    int binary;
    binary = (data >> shift) & 1;
    return binary;
}

int main()
{
    printf("Train a small ANN to the XOR function using backpropagation.\n");
    srand(time(0));
    //初始化bit長度 訓練次數 問題數 hidden layers數 和神經元數
    int bitlong = 4;
    int trainingtime = 1000;
    int problemnum = 4;
    int hidden = 1;
    int neurons = 2;

    printf("Please input the number of bit:");
    scanf("%d", &bitlong);
    printf("Please input how many times you want to train:");
    scanf("%d", &trainingtime);
    printf("Please input how many problem you want to test(the problem will generate automatically):");
    scanf("%d", &problemnum);
    printf("Please input how many hidden layers:");
    scanf("%d", &hidden);
    printf("Please input how many neurons:");
    scanf("%d", &neurons);

    int* problemdata = (int*)malloc(sizeof(int) * problemnum);
    int* trainingdata = (int*)malloc(sizeof(int) * trainingtime);
    int** binarytrainingdata = (int**)malloc(sizeof(int*) * trainingtime);

    for (int i = 0; i < trainingtime; i++) {
        int** m = binarytrainingdata + i;
        *m = (int*)malloc(bitlong * sizeof(int));
    }

    int** binaryproblemdata = (int**)malloc(sizeof(int*) * problemnum);
    for (int i = 0; i < problemnum; i++) {
        int** l = binaryproblemdata + i;
        *l = (int*)malloc(bitlong * sizeof(int));
    }

    for (int i = 0; i < problemnum; ++i) {//生成題目 同時避免生成重複題目
        problemdata[i] = summondata();
        for (int j = 0; j < i; ++j) {
            if (problemdata[i] == problemdata[j]) {
                i--;
                break;
            }
        }
    }

    for (int i = 0; i < trainingtime; ++i) {//生成訓練資料 同時避免訓練資料與題目重複
        trainingdata[i] = summondata();
        for (int j = 0; j < bitlong; ++j) {
            if (trainingdata[i] == problemdata[j]) {
                i--;
                break;
            }
        }
    }

    double** problemdatacopy = (double**)malloc(sizeof(double) * problemnum);
    double** trainingdatacopy = (double**)malloc(sizeof(double) * trainingtime);
    for (int i = 0; i < problemnum; i++)
        problemdatacopy[i] = (double*)malloc(bitlong * sizeof(double));
    for (int i = 0; i < trainingtime; i++)
        trainingdatacopy[i] = (double*)malloc(bitlong * sizeof(double));

    int tmp = 0;

    for (int i = 0; i < problemnum; ++i) {//將資料轉換成二進制
        for (int j = 0; j < bitlong; ++j) {
            binaryproblemdata[i][j] = binarydata(problemdata[i], j);
        }
    }

    for (int i = 0; i < trainingtime; ++i) {
        for (int j = 0; j < bitlong; ++j) {
            binarytrainingdata[i][j] = binarydata(trainingdata[i], j);
        }
    }
    double* problemoutput = (double*)malloc(sizeof(double) * problemnum);
    double* trainingoutput = (double*)malloc(sizeof(double) * trainingtime);
    for (int i = 0; i < trainingtime; ++i) {//計算訓練資料的答案
        for (int j = 0; j < bitlong; ++j) {
            tmp += binarytrainingdata[i][j];
        }
        tmp %= 2;
        //printf("%d\n", tmp);
        if (tmp == 0) {
            trainingoutput[i] = 1;
        }
        else {
            trainingoutput[i] = 0;
        }
        tmp = 0;
    }

    

    for (int i = 0; i < problemnum; ++i) {//將資料由int轉換成double
        for (int j = 0; j < bitlong; ++j) {
            problemdatacopy[i][j] = (double)binaryproblemdata[i][j];
        }
    }
    for (int i = 0; i < trainingtime; ++i) {
        for (int j = 0; j < bitlong; ++j) {
            trainingdatacopy[i][j] = (double)binarytrainingdata[i][j];
        }
    }
    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */ 
    genann* ann = genann_init(2, hidden, neurons, 1);
    /* Train on the four labeled data points many times. */
    for (int i = 0; i < trainingtime; ++i) {
        genann_train(ann, trainingdatacopy[i], trainingoutput + i, 3);
    }
    /* Run the network and see what it predicts. */
    for (int i = 0; i < problemnum; ++i) {
        printf("Output for [");
        for (int j = 0; j < bitlong; ++j) {
            printf("%1.f, ", problemdatacopy[i][j]);
        }
        printf("] is %1.f.\n", *genann_run(ann, problemdatacopy[i]));
    }
  

    genann_free(ann);
    free(problemdata);
    free(trainingdata);
    free(binaryproblemdata);
    free(binarytrainingdata);
    free(problemoutput);
    free(trainingoutput);
    free(problemdatacopy);
    free(trainingdatacopy);
    system("pause");
    return 0;
}
