#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <iostream>
#include <vector>
#include <set>
#include <math.h>
#include <string>
#include <algorithm>
using namespace std;
typedef long long ll; typedef unsigned long long ull;
#define REP(i,n) for (int i=0;i<n;i++)
#define FOR(i,a,b) for (int i=a;i<b;i++)
void array_pop(int a[], int& n, int pos) {
	FOR(i, pos, n - 1) a[i] = a[i + 1];
	n--;
}
#define INF 2*100000+4
#define MAX 1000
int a[] = {4,5,3,2,6};
int flag[MAX];
int f[MAX];
int main()
{
	int n = sizeof(a) / sizeof(a[0]);
	REP(i, n) flag[i] = true;
	REP(i, n) {
		f[i] = f[i - 1] + a[i];
		if (a[i] != 1 && a[i - 1] != 1 && !(a[i] == 2 && a[i - 1] == 2))
			if (f[i] < f[i - 2] + a[i - 1] * a[i]) {
				f[i] = f[i - 2] + a[i - 1] * a[i];
				flag[i] = false; flag[i - 1] = true;
			}
	}
	cout <<"Result: "<< f[n - 1]<<endl<<"Checking: ";
	REP(i, n) if (!flag[i]) {
		a[i-1] *= a[i];
	}
	int sum = 0;
	REP(i, n) {
		cout << a[i] << " "; if (flag[i]) sum += a[i]; cout << flag[i] << endl;
	}
	cout << endl << sum;
	return 0;
}