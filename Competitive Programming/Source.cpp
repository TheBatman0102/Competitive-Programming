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
#define INF 2*100000+4
#define MAX 1000
using namespace std;
string s1, s2;
int d[MAX][MAX];
int main()
{
	s1 = "ALTRUISTIC"; s2 = "ALGORITHM";
	int s1n = s1.length(), s2n = s2.length();
	REP(i, s1n + 1) d[i][0] = i;
	REP(i, s2n + 1) d[0][i] = i;
	FOR(i, 1, s1n+1) FOR(j, 1, s2n+1)
		d[i][j] = min(d[i - 1][j]+1, min(d[i][j - 1]+1, d[i - 1][j - 1] + (s1[i-1] != s2[j-1])));
	cout << "    ";
	REP(i, s1n) cout << s1[i] << " ";
	cout << endl;
	REP(j, s2n+1) {
		if (1 <= j) cout << s2[j - 1] << " ";
		else cout << "  ";
		REP(i, s1n + 1) cout << d[i][j] << " ";
		cout << endl;
	}
	//cout << (s1[1]!=s2[1]);
	return 0;
}