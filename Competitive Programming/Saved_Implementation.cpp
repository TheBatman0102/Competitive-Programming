#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
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

#define forn(i, n) for (int i = 0; i < n; i++)
#define forab(i,a,b) for (int i=a;i<b;i++)
#define eps 1e-9
#define MOD (ll)((1e9)+7)

using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef vector<int> vi;
typedef vector<ll> vll;
typedef set<int> si;

int gcd(int a, int b) {
    return b ? gcd(b, a % b) : a;
}
int lcm(int a, int b) {
    return a * b / gcd(a, b);
}

int MinQuerries()
{
    vi a{ 1,3,4,8,6,1,4,2 };
    int n = a.size();
    int max2pwr = 1;
    while (max2pwr * 2 <= n) {
        max2pwr <<= 1;
    }
    vi save2pwr; int temp = 1;
    vector<vector<int>> minArr(n, vi(n));
    forn(i, n) {
        minArr[i][i] = a[i];
        if (temp * 2 == i + 1)
            temp *= 2;
        save2pwr.push_back(temp);
    }
    for (int l = 2; l <= max2pwr; l *= 2)
        for (int start = 0; start < n - l + 1; start++) {
            int end = start + l - 1; int half = l / 2;
            minArr[start][end] = min(minArr[start][start + half - 1], minArr[start + half][end]);
        }
    int start = 2, end = 4;
    int subInd = save2pwr[end - start];
    cout << min(minArr[start][start + subInd - 1], minArr[end - subInd + 1][end]);

    return 0;
}

namespace FenwickTree {
    int getSum(int BITree[], int index)
    {
        int sum = 0; // Iniialize result

        // index in BITree[] is 1 more than the index in arr[]
        index = index + 1;

        // Traverse ancestors of BITree[index]
        while (index > 0)
        {
            // Add current element of BITree to sum
            sum += BITree[index];

            // Move index to parent node in getSum View
            index -= index & (-index);
        }
        return sum;
    }

    void updateBIT(int BITree[], int n, int index, int val)
    {
        // index in BITree[] is 1 more than the index in arr[]
        index = index + 1;

        // Traverse all ancestors and add 'val'
        while (index <= n)
        {
            // Add 'val' to current node of BI Tree
            BITree[index] += val;

            // Update index to that of parent in update View
            index += index & (-index);
        }
    }

    int* constructBITree(int a[], int n) {
        int* BITree = new int[n + 1];
        forn(i, n) BITree[i + 1] = 0;
        forn(i, n) updateBIT(BITree, n, i, a[i]);
        return BITree;
    }

    int Fenwick()
    {
        int freq[] = { 2, 1, 1, 3, 2, 3, 4, 5, 6, 7, 8, 9 };
        int n = sizeof(freq) / sizeof(freq[0]);
        int* BITree = constructBITree(freq, n);
        cout << "Sum of elements in arr[0..5] is "
            << getSum(BITree, 5);

        // Let use test the update operation
        freq[3] += 6;
        updateBIT(BITree, n, 3, 6); //Update BIT for above change in arr[]

        cout << "\nSum of elements in arr[0..5] after update is "
            << getSum(BITree, 5);

        return 0;
    }
}

class SumSegmentTree {
public:
    vector<int> seg;
    int checkpoint;
    SumSegmentTree(vector<int>& nums) {
        int lim = 1, n = nums.size();
        while (lim < n) lim <<= 1;
        checkpoint = lim;
        seg.resize(2 * lim);
        for (int i = 0; i < n; i++) seg[i + lim] = nums[i];
        for (int i = lim - 1; i >= 1; i--)
            seg[i] += seg[(i << 1) + 1] + seg[i << 1];

    }

    void update(int index, int val) {
        int realInd = index + checkpoint;
        int dif = val - seg[realInd];
        while (realInd >= 1) {
            seg[realInd] += dif;
            realInd >>= 1;
        }
    }

    int sumRange(int left, int right) {
        int rleft = left + checkpoint, rright = right + checkpoint;
        int res = 0;
        while (rleft <= rright) {
            if ((rleft & 1) == 1) res += seg[rleft++];
            if ((rright & 1) == 0) res += seg[rright--];
            rleft >>= 1; rright >>= 1;
        }
        return res;
    }
    /**
     * Your NumArray object will be instantiated and called as such:
     * NumArray* obj = new NumArray(nums);
     * obj->update(index,val);
     * int param_2 = obj->sumRange(left,right);
     */
};

class MinSegmentTree {
public:
    vector<int> seg;
    int checkpoint;
    MinSegmentTree(vector<int>& nums) {
        int lim = 1, n = nums.size();
        while (lim < n) lim <<= 1;
        checkpoint = lim;
        seg.resize(2 * lim);
        for (int i = 0; i < n; i++) seg[i + lim] = nums[i];
        for (int i = lim - 1; i >= 1; i--)
            seg[i] = min(seg[(i << 1) + 1], seg[i << 1]);

    }

    void update(int index, int val) {
        int realInd = index + checkpoint;
        realInd >>= 1;
        seg[realInd] = val;
        while (realInd >= 1) {
            seg[realInd] = min(seg[realInd << 1], seg[(realInd << 1) + 1]);
            realInd >>= 1;
        }
    }

    int minRange(int left, int right) {
        int rleft = left + checkpoint, rright = right + checkpoint;
        int res = seg[rleft];
        while (rleft <= rright) {
            if ((rleft & 1) == 1) res = min(res, seg[rleft++]);
            if ((rright & 1) == 0) res = min(res, seg[rright--]);
            rleft >>= 1; rright >>= 1;
        }
        return res;
    }
    /**
     * Your NumArray object will be instantiated and called as such:
     * NumArray* obj = new NumArray(nums);
     * obj->update(index,val);
     * int param_2 = obj->sumRange(left,right);
     */
};

void HanoiTower(int n, char a, char b, char c) {
    if (n == 1) {
        cout << "\t" << a << "-------" << c << endl;
        return;
    }
    HanoiTower(n - 1, a, c, b);
    HanoiTower(1, a, b, c);
    HanoiTower(n - 1, b, a, c);
}

class BasicGraph {
    int N;
    vector<vector<int>> adj;
    vector<bool> visited;
    public:
        BasicGraph() { N = 0; }
        BasicGraph(int n) {
            N = n;
            adj.resize(N);
            visited.resize(N);
        }
        void setSize(int n) {
            N = n;
            adj.resize(N);
            visited.resize(N);
        }
        void addEdge(int x, int y) {
            N++;
            adj[x].push_back(y); adj[y].push_back(x);
        }
        void DFS(int root) {
            if (visited[root]) return;
            visited[root] = true;
            //process node
            cout << root << " ";
            //end
            for (int node : adj[root])
                DFS(node);
        }
        void BFS(int root) {
            queue<int> q;
            q.push(root);
            visited[root] = true;
            while (!q.empty()) {
                int node = q.front(); q.pop();
                //process node
                cout << node << " ";
                //end
                for (int child : adj[node]) {
                    if (visited[child]) continue;
                    visited[child] = true;
                    q.push(child);
                }
            }
        }

};

namespace DIJKSTRA {
    int n = 5;
    vector <vector<pair<int, int>>> adj = {
            {{0,0}},
            {{5,2},{9,4},{1,5}},
            {{2,3},{5,1}},
            {{2,2},{6,4}},
            {{6,3},{9,1},{2,5}},
            {{2,4},{1,1}}
    };
    vector<int> dis(n + 1, INT32_MAX);
    vector<bool> processed(n + 1, false);

    void dijkstra(int root) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>> > q;
        dis[root] = 0; q.push({ dis[root],root });
        while (!q.empty()) {
            auto node = q.top(); q.pop();
            if (processed[node.second]) continue;
            processed[node.second] = true;
            for (auto child : adj[node.second]) {
                if (processed[child.second]) continue;
                dis[child.second] = min(dis[child.second], dis[node.second] + child.first);
                q.push({ dis[child.second],child.second });
            }
        }
    }
    void main()
    {
        int root = 1;
        dijkstra(root);
        cout << "The distance from " << root << " to the following points are : " << endl;
        forab(i, 1, n + 1)
            cout << "(" << i << "): " << dis[i] << endl;

    }
}

class Tree {
    int size;
    vector<vi> adj;
    vi subtreeSizeRoot;
public:
    Tree(int n) {
        size = 0;
        adj.resize(n + 1);
        subtreeSizeRoot.resize(n + 1);
    }
    void addNode(int x, int y) {
        size++;
        adj[x].push_back(y);
        adj[y].push_back(x);
    }
    void dfs(int root, int prev) {
        //process root
        cout << root << " ";
        for (int node : adj[root])
            if (node != prev) dfs(node, root);
    }
    int getSize() {
        return size;
    }
    void subtreeSizeDFS(int root, int prev) {
        subtreeSizeRoot[root] = 1;
        for (int node : adj[root]) {
            if (node == prev) continue;
            subtreeSizeDFS(node, root);
            subtreeSizeRoot[root] += subtreeSizeRoot[node];
        }
    }
    int subtreeSize(int root, int prev) {
        subtreeSizeDFS(root, prev);
        return subtreeSizeRoot[root];
    }
    //calculate the longest path from a node to a leaf
    pair<int, int> toLeaf(int root, int prev = 0) {
        pair<int, int> res = { 0,root };
        for (int child : adj[root]) {
            if (child == prev) continue;
            auto childres = toLeaf(child, root);
            childres.first++;
            res = max(res, childres);
        }
        return res;
    }
    void findDiameter(int root) {
        int dis, endpoint1, endpoint2;
        endpoint1 = toLeaf(root).second;
        auto pair2 = toLeaf(endpoint1);
        dis = pair2.first; endpoint2 = pair2.second;
        cout << "The diameter is " << dis << " (path " << endpoint1 << " -> " << endpoint2 << ")" << endl;
    }
};

class DisjoinSets {
    vi link;
    vi size;

    DisjoinSets(int n) {
        link.resize(n + 1);
        size.resize(n + 1);
        forab(i, 1, n + 1) {
            link[i] = i;
            size[i] = 1;
        }
    }

    int find(int node) { //find the component to which node belong
        while (link[node] != node) node = link[node];
        return node;
    }

    //2 functions used in Kruskal's algorithm
    bool same(int node1, int node2) {
        return (find(node1) == find(node2));
    }

    void unite(int node1, int node2) {
        int a = find(node1), b = find(node2);
        if (size[a] < size[b]) swap(a, b);
        size[a] += size[b];
        link[b] = a;
    }
};

namespace Kruskal {
    class DisjoinSets {
        vi link;
        vi size;
        public:
            DisjoinSets(int n) {
                link.resize(n + 1);
                size.resize(n + 1);
                forab(i, 1, n + 1) {
                    link[i] = i;
                    size[i] = 1;
                }
            }

            int find(int node) { //find the component to which node belong
                while (link[node] != node) node = link[node];
                return node;
            }

            //2 functions used in Kruskal's algorithm
            bool same(int node1, int node2) {
                return (find(node1) == find(node2));
            }

            void unite(int node1, int node2) {
                int a = find(node1), b = find(node2);
                if (size[a] < size[b]) swap(a, b);
                size[a] += size[b];
                link[b] = a;
            }
    };

    vector<vector<pair<int, int>>> adj = {
        {{0,0}},
        {{3,2},{5,5}},
        {{3,1},{5,3},{6,5}},
        {{5,2},{9,4},{3,6}},
        {{9,3},{7,6}},
        {{5,1},{6,2},{2,6}},
        {{3,3},{7,4},{2,5}}
    };

    void main() {
        vector<tuple<int, int, int>> q;
        int graphSize = 6;
        forab(i, 1, graphSize + 1)
            for (auto edge : adj[i]) q.push_back({ edge.first,i,edge.second });
        sort(q.begin(), q.end());
        DisjoinSets set(graphSize);
        vector<pair<int, int>> res;
        for (auto edge : q) {
            int length, node1, node2;
            tie(length, node1, node2) = edge;
            if (!set.same(node1, node2)) {
                res.push_back({ node1,node2 });
                set.unite(node1, node2);
            }
        }
        cout << "The edge of the spanning tree is:" << endl;
        for (auto edge : res) cout << "{" << edge.first << "," << edge.second << "} ; ";
    }
}

namespace Prim {
    vector<vector<pair<int, int>>> adj = {
    {{0,0}},
    {{3,2},{5,5}},
    {{3,1},{5,3},{6,5}},
    {{5,2},{9,4},{3,6}},
    {{9,3},{7,6}},
    {{5,1},{6,2},{2,6}},
    {{3,3},{7,4},{2,5}}
    };

    void main() {
        priority_queue<tuple<int, int, int>> q;
        int graphSize = 6;
        int root = 1;
        vector<bool> visited(graphSize + 1, false);
        vector<pair<int, int>> res;
        for (auto edge : adj[root]) q.push({ -edge.first, root, edge.second });
        visited[root] = true;
        while (!q.empty()) {
            auto minEdge = q.top(); q.pop();
            int length, node1, node2; tie(length, node1, node2) = minEdge;
            if (visited[node2]) continue;
            visited[node2] = true;
            res.push_back({ node1,node2 });
            for (auto edge : adj[node2])
                if (!visited[edge.second]) q.push({ -edge.first,node2,edge.second });
        }
        cout << "Starting point of Prim's algorithm: " << root << endl;
        cout << "The edge of the spanning tree is:" << endl;
        for (auto edge : res) cout << "{" << edge.first << "," << edge.second << "} ; ";
    }
}

int binarySearch(vector<int> nums, int target) {
    int n = nums.size();
    int l = 0, r = n - 1, mid;
    while (l <= r) {
        mid = l + (r - l) / 2;
        if (target == nums[mid])
            return mid;
        if (target < nums[mid])
            r = mid - 1;
        else
            l = mid + 1;
    }
    return -1;
}

void mergeSort(int a[], int l, int r) {
    if (l >= r) return;
    int mid = l + (r - l) / 2;
    mergeSort(a, l, mid);
    mergeSort(a, mid + 1, r);

    // merge sorted subarray
    int* leftArr = new int[mid - l + 1];
    int* rightArr = new int[r - mid];
    forn(i, mid - l + 1) leftArr[i] = a[l + i];
    forn(i, r - mid) rightArr[i] = a[mid + 1 + i];
    int i = 0, j = 0, cur = l;
    while (i < mid - l + 1 || j < r - mid) {
        if (i >= mid - l + 1) a[cur++] = rightArr[j++];
        else if (j >= r - mid) a[cur++] = leftArr[i++];
        else if (leftArr[i] < rightArr[j]) a[cur++] = leftArr[i++];
        else a[cur++] = rightArr[j++];
    }
}

namespace BST {
    struct Node {
        int val;
        Node* left, * right;

        Node(int v) : val(v), left(nullptr), right(nullptr) {}
        ~Node() {
            if (left != nullptr) delete left;
            if (right != nullptr) delete right;
        }
    };

    class BST {
    private:
        Node* root;
    public:
        BST() : root(nullptr) {}
        ~BST() {
            if (root != nullptr)
                delete root; // traverses through destructors
        }

        Node* Insert(int x, Node* parent) {
            Node* result = parent;

            if (parent == nullptr) {
                result = new Node(x);
            }
            else if (x < parent->val) {
                parent->left = Insert(x, parent->left);
            }
            else if (x > parent->val) {
                parent->right = Insert(x, parent->right);
            }
            else {
                return nullptr; // duplicate, add should not comply
            }

            return result;
        }

        bool Insert(int x) {
            if (Find(x) != nullptr)
                return false;

            root = Insert(x, root);
            return true;
        }

        Node* Find(int x, Node* parent) {
            Node* current = parent;
            int currentValue;

            while (current != nullptr) {
                currentValue = current->val;
                if (x < currentValue) {
                    current = current->left;
                }
                else if (x > currentValue) {
                    current = current->right;
                }
                else {
                    return current;
                }
            }

            return nullptr;
        }

        Node* Find(int x) {
            if (root != nullptr)
                return Find(x, root);

            return nullptr;
        }

        Node* FindMin(Node* parent) {
            Node* current = parent;
            Node* left = nullptr;

            if (current != nullptr) {
                while ((left = current->left) != nullptr)
                    current = left;
            }

            return current;
        }

        Node* RemoveMin(Node* parent) {
            if (parent == nullptr) {
                return nullptr;
            }
            else if (parent->left != nullptr) {
                parent->left = RemoveMin(parent->left);
                return parent;
            }
            else {
                Node* result = parent->right;

                parent->right = parent->left = nullptr;
                delete parent;
                return result;
            }
        }

        Node* Remove(int x, Node* parent) {
            Node* current = parent;
            Node* left = nullptr;
            Node* right = nullptr;
            int currentValue;

            if (current != nullptr) {
                left = current->left;
                right = current->right;
                currentValue = current->val;
            }

            if (current == nullptr) {
                return nullptr;
            }
            else if (x < currentValue) {
                current->left = Remove(x, left);
            }
            else if (x > currentValue) {
                current->right = Remove(x, right);
            }
            else if (left != nullptr && right != nullptr) {
                current->val = FindMin(right)->val;
                current->right = RemoveMin(right);
            }
            else {
                current = (left != nullptr) ? left : right;

                parent->right = parent->left = nullptr;
                delete parent;
            }

            return current;
        }

        bool Remove(int x) {
            if (Find(x) == nullptr)
                return false;

            root = Remove(x, root);
            return true;
        }
    };
}

// find the longest strictly increasing subsequence of an array in O(nlogn)
namespace LongestIncreasingSubsequence {
    int lengthOfLIS(vector<int> nums) {
        int n = nums.size();
        vector<int> tail(n);
        int res = 0;
        for (int num : nums) {
            if (res == 0) {
                tail[0] = num;
                res++;
                continue;
            }
            int i = 0, j = res - 1;
            while (i < j) {
                int mid = i + (j - i) / 2;
                if (num > tail[mid])
                    i = mid + 1;
                else
                    j = mid;
            }
            if (i == res - 1 && num > tail[i]) {
                tail[res++] = num;
                continue;
            }
            tail[i] = num;
        }
        return res;
    }
}

namespace Trie {
    struct TrieNode {
        bool isEndOfWord;
        TrieNode* child[26];
    };

    TrieNode* getNode() {
        TrieNode* res = new TrieNode;
        res->isEndOfWord = false;
        forn(i, 26) res->child[i] = NULL;
        return res;
    }

    void insert(TrieNode* root, string word) {
        int n = word.size();
        TrieNode* current = root;
        forn(i, n) {
            int index = word[i] - 'a';
            if (current->child[index] == NULL) current->child[index] = getNode();
            current = current->child[index];
        }
        current->isEndOfWord = true;
    }

    bool search(TrieNode* root, string word) {
        int n = word.size();
        auto current = root;
        forn(i, n) {
            int index = word[i] - 'a';
            if (current->child[index] == NULL) return false;
            current = current->child[index];
        }
        return true;
    }
}

int EggDrop(int trials, int eggs) {
    // return the largest number of floor that can be covered using "trials" trials and "eggs" eggs
    int sum = 0, current = 1;
    forab(i,1, eggs+1) {
        current *= (trials - i + 1);
        current /= i;
        sum += current;
    }
    return sum;
}