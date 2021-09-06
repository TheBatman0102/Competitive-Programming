#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <math.h>
#include <string>
#include <algorithm>
#include <map>
#include <bitset>
#include <deque>
#include <stack>
#include <queue>
#include <unordered_map>
#include <numeric>
using namespace std;

#define forn(i, n) for (int i = 0; i < n; i++)
#define forab(i,a,b) for (int i=a;i<b;i++)
#define eps 1e-9

typedef long long ll;
typedef unsigned long long ull;

void main() {
	ios_base::sync_with_stdio(false);

	int t; cin >> t;
	forn(_, t) {
		int n; cin >> n;
		vector<int> a(n+1), b(n+1);
		forab(i,1, 1+n)
			cin >> a[i];
		vector<int> giamInd, tangInd;
		int tongGiam=0, tongTang=0;
		forab(i,1,1+ n) {
			cin >> b[i];
			if (a[i] > b[i]) {
				giamInd.push_back(i);
				tongGiam += a[i] - b[i];
			}
			else if (a[i] < b[i]) {
				tangInd.push_back(i);
				tongTang += b[i] - a[i];
			}
		}
		if (tongGiam != tongTang) {
			cout << -1 << endl;
			continue;
		}
		if (giamInd.empty()) {
			cout << 0 << endl;
			continue;
		}
		cout << tongGiam << endl;
		int curTang = 0;
		for (int i : giamInd) {

			while (a[i] > b[i]) {
				cout << i << " " << tangInd[curTang] << endl;
				a[i]--; a[tangInd[curTang]]++;
				if (a[tangInd[curTang]] == b[tangInd[curTang]])
					curTang++;
			}
		}

	}
}