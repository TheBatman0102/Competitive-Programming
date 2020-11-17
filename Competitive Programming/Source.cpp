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
int p[] = {3,5,7};
int e[] = {8,12,4};
int f[MAX][MAX];
int win = 0;
int minp;
int main()
{
	int n = sizeof(p) / sizeof(p[0]);
	REP(i, n) win += e[i];
	REP(j, win + 1) if (j == e[0]) f[0][j] = p[0] / 2 + 1;
	else f[0][j] = INF;
	REP(i, n) f[i][0] = 0;
	FOR(i,1, n) REP(j, win+1)
	{
		f[i][j] = f[i - 1][j];
		if (e[i] <= j) f[i][j] = min(f[i][j], f[i - 1][j - e[i]] + p[i]/2+1);
	}
	minp = f[n - 1][win / 2 + 1];
	FOR(j, win / 2 + 2, win + 1) minp = min(minp, f[n - 1][j]);
	//check
	REP(i, n) cout << e[i]<<"("<<p[i] / 2 + 1<<") "; cout << endl;
	cout << "Total electoral votes: " << win << endl << "Votes to win: " << win / 2 + 1 << endl;
	cout << minp << endl;
	//REP(i, n) REP(j, win + 1)  cout <<i<<" "<<j<<" "<< f[i][j] << endl;
	return 0;
}