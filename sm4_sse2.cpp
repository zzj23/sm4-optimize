#include<iostream>
#include<Windows.h>
#include <emmintrin.h>

const unsigned int nElements = 4;
const unsigned int N = 40;
using namespace std;

typedef double it;

void winograd_acc(double* A, double* B, double* C, int M, int P, int N, int strideA, int strideB, int strideC);
void liner_real_mat(double* a, double* b, double* c);
void vec_add(double* a, double* b, double* c);
float for_add(double* a);

template<class T>
void measure(T&& func) {//能同时处理4个字

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);
	func();
	QueryPerformanceCounter(&t2);
	double time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
	cout << "time = " << time << "s" << endl;  //输出时间（单位：ｓ）

}

__m128d A1, B1, C1, tmp;

double* a = (double*)_mm_malloc(N*N * sizeof(double), 16);
double* b = (double*)_mm_malloc(N*N * sizeof(double), 16);
double* c = (double*)_mm_malloc(N*N * sizeof(double), 16);

int main() {
    for (int i = 0; i < N*N;  i++) {
        a[i] = 1;
        b[i] = 1;
    }

	measure([]() {
        liner_real_mat(a,b,c);
		});

   // for (int i = 0; i < N; i++) {
   //     for (int j = 0; j < N; j++)
   //     {
   //        cout << c[i] << " ";
   //     }
   //     cout << endl;
   // }

	_mm_free(a);
	_mm_free(b);
	_mm_free(c);
}

void liner_real_mat(double*a, double*b, double*c) {
    double* tmp2 = (double*)_mm_malloc(N * sizeof(double), 16);
    double x;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k+=4) {
				A1 = _mm_loadu_pd(&a[i * N + k]);
				B1 = _mm_loadu_pd(&b[j * N + k]);
				tmp = _mm_mul_pd(A1, B1);
                _mm_store_sd(&tmp2[k],tmp);
			}
            c[i * N + j] = for_add(tmp2);
		}
	}

	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < N; j++) {
	//		cout << c[j + i] << " ";
	//	}
	//	cout << endl;
	//}

}


float for_add(double* a) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

void vec_add(double* a, double* b, double* c) {
	for (unsigned int i = 0; i < nElements; i+= 4) {
		A1 = _mm_loadu_pd(&a[i]);
		B1 = _mm_loadu_pd(&b[i]);
		C1 = _mm_mul_pd(A1, B1);
        _mm_store_sd(&c[i], C1);
	}


	//for (int i = 0; i < nElements; i++) {
	//	cout << c[i] << endl;
	//}
}

void winograd_acc(double* A, double* B, double* C, int M, int P, int N, int strideA, int strideB, int strideC) {
    if ((M <= 64) || (M % 2 != 0 || N % 2 != 0 || P % 2 != 0))
    {
        return liner_real_mat(A,B,C);//, strideA, strideB, strideC);
    }

    /*initial S*/
    it* S1 = new it[(M / 2) * (P / 2)];
    it* S2 = new it[(M / 2) * (P / 2)];
    it* S3 = new it[(M / 2) * (P / 2)];
    it* S4 = new it[(M / 2) * (P / 2)];
    int id_A = 0, offset_a = 0, id_S = 0;
    for (int i = 0; i < M / 2; i++) {
        for (int j = 0; j < P / 2; j++) {
            id_S = i * (P / 2) + j;
            /*S1*/
            id_A = (i + (M / 2)) * strideA + j;
            offset_a = P / 2;
            S1[id_S] = A[id_A] + A[M + offset_a];
            /*S2*/
            id_A = i * strideA + j;
            S2[id_S] = S1[id_S] - A[id_A + offset_a];
            /*S3*/
            offset_a = (M / 2) * strideA;
            S3[id_S] = A[id_A] - A[id_A + offset_a];
            /*S4*/
            id_A = i * strideA + (P / 2) + j;
            S4[id_S] = A[id_A] - S2[id_S];
        }
    }

    /*initial T*/
    it* T1 = new it[(P / 2) * (N / 2)];
    it* T2 = new it[(P / 2) * (N / 2)];
    it* T3 = new it[(P / 2) * (N / 2)];
    it* T4 = new it[(P / 2) * (N / 2)];

    int id_B = 0, offset_b = 0, id_T = 0;
    for (int i = 0; i < P / 2; i++) {
        for (int j = 0; j < N / 2; j++) {
            id_T = i * (N / 2) + j;
            /*T1*/
            id_B = i * strideB + j;
            offset_b = N / 2;
            T1[id_T] = B[id_B + offset_b] - B[id_B];
            /*T2*/
            id_B = (i + (P / 2)) * strideB + (N / 2) + j;
            T2[id_T] = B[id_B] - T1[id_T];
            /*T3*/
            id_B = i * strideB + (N / 2) + j;
            offset_b = (P / 2) * strideB;
            T3[id_T] = B[id_B + offset_b] - B[id_B];
            /*T4*/
            id_B = (i + (P / 2)) * strideB + j;
            T4[id_T] = T2[id_T] - B[id_B];
        }
    }
    /*initial M*/
    it* M1 = new it[(M / 2) * (N / 2)];
    it* M2 = new it[(M / 2) * (N / 2)];
    it* M3 = new it[(M / 2) * (N / 2)];
    it* M4 = new it[(M / 2) * (N / 2)];
    it* M5 = new it[(M / 2) * (N / 2)];
    it* M6 = new it[(M / 2) * (N / 2)];
    it* M7 = new it[(M / 2) * (N / 2)];
    winograd_acc(A, B, &M1[0], M / 2, P / 2, N / 2, strideA, strideB, N / 2);
    winograd_acc(&A[P / 2], &B[(P / 2) * strideB], &M2[0], M / 2, P / 2, N / 2, strideA, strideB, N / 2);
    winograd_acc(&S4[0], &B[(P / 2) * strideB + (N / 2)], &M3[0], M / 2, P / 2, N / 2, P / 2, strideB, N / 2);
    winograd_acc(&A[(M / 2) * strideA + (P / 2)], &T4[0], &M4[0], M / 2, P / 2, N / 2, strideA, N / 2, N / 2);
    winograd_acc(&S1[0], &T1[0], &M5[0], M / 2, P / 2, N / 2, P / 2, N / 2, N / 2);
    winograd_acc(&S2[0], &T2[0], &M6[0], M / 2, P / 2, N / 2, P / 2, N / 2, N / 2);
    winograd_acc(&S3[0], &T3[0], &M7[0], M / 2, P / 2, N / 2, P / 2, N / 2, N / 2);
    /*initial C*/
    int id_M = 0;
    for (int i = 0; i < i < M / 2; i++) {
        for (int j = 0; j < N / 2; j++) {
            id_M = i * (N / 2) + j;
            C[i * strideC + j] = M1[id_M] + M2[id_M];
            C[i * strideC + j + (N / 2)] = M1[id_M] + M6[id_M] + M5[id_M] + M3[id_M];
            C[(i + (M / 2)) * strideC + j] = M1[id_M] + M6[id_M] + M7[id_M] - M4[id_M];
            C[(i + (M / 2)) * strideC + j + (N / 2)] = M1[id_M] + M6[id_M] + M7[id_M] + M5[id_M];
        }
    }
    delete[]S1;
    S1 = nullptr;
    delete[]S2;
    S2 = nullptr;
    delete[]S3;
    delete[]S4;
    S3 = nullptr;
    S4 = nullptr;
    delete[]T1;
    T1 = nullptr;
    delete[]T2;
    T2 = nullptr;
    delete[]T3;
    T3 = nullptr;
    delete[]T4;
    T4 = nullptr;

    delete[]M1;
    M1 = nullptr;
    delete[]M1;
    M2 = nullptr;
    delete[]M3;
    M3 = nullptr;
    delete[]M4;
    M4 = nullptr;
    delete[]M5;
    M5 = nullptr;
    delete[]M6;
    M6 = nullptr;
    delete[]M7;
    M7 = nullptr;
}

