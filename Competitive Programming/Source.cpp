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
int a[] = { 7,5,3,6,2,2,2,5,1,5,2,2,2,1,8 };
int oflag[MAX], tflag[MAX], cflag[MAX];
int f[MAX];
int b[MAX];
int main()
{
	int n = sizeof(a) / sizeof(a[0]);
	REP(i, n) cout << a[i] << " ";
	cout << endl;
	REP(i, n) { oflag[i] = true; tflag[i] = true; cflag[i] = true; }
	f[0] = a[0];
	if (n > 1) {
		f[1] = max(a[0] + a[1], a[0] * a[1]);
		if (f[1] > a[0] + a[1]) cflag[1] = false;
	}
	FOR(i, 2, n) {
		tflag[i - 2] = cflag[i - 2]; tflag[i - 1] = cflag[i - 1]; tflag[i - 3] = cflag[i - 3];
		f[i] = max(f[i - 1] + a[i], f[i - 2] + a[i - 1] * a[i]);
		if (f[i] > f[i - 1] + a[i]) {
			cflag[i] = false; cflag[i - 1] = true; cflag[i - 2] = oflag[i - 2]; cflag[i - 3] = oflag[i - 3];
		}
		oflag[i - 2] = tflag[i - 2]; oflag[i - 1] = tflag[i - 1]; oflag[i - 3] = tflag[i - 3];
	}
	int res = f[n - 1];
	cout << "Result: " << res << endl << "Checking: ";
	int m = 0;
	REP(i, n) {
		if (cflag[i]) b[m] = a[i]; else b[--m] *= a[i];
		m++;
	}
	int sum = 0;
	REP(i, m) {
		cout << b[i] << " "; sum += b[i];
	}
	cout << endl << boolalpha << (sum == res);
	return 0;
}