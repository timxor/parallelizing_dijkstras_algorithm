/*
***Correctness***                   
Program Compiles    5   /   5       
Read in input in corerect format    5   /   5       
Distributes the matrix by columns   5   /   5       
Allocates correct amount of storage for loc_dist, loc_pred and loc_known: loc_n elements    3   /   3       
Correctly intialize loc_dist, loc_pred and loc_known    3   /   3       
Correctly finds the "nearest" vertex to 0   5   /   5       
Correctly updates loc_dist and loc_pred 5   /   5       
Prints the distances correctly  3   /   3       
Prints the paths correctly  3   /   3       
Free dynamically allocated memory and the MPI datatype  3   /   3       
                    
Test mat6 1 process 5   /   5       
Test mat6 2 processes   5   /   5       
Test mat6 3 processes   5   /   5       
                    
Test mat16 1 process    5   /   5       
Test mat16 2 processes  5   /   5       
Test mat16 4 processes  5   /   5       
Test mat16 8 processes  5   /   5       
                    
###Total Correctness###     75  /   75  
                    
***Documentation***                 
Purpose of program and functions    3   /   3
Explanation of program input and output 2   /   2
Explanation of function args and return value   3   /   3       
Explanation of obscure constructs   2   /   2       
###Total Documentation###       10   /   10  
                    
***Source format***                 
Consistent capitalization       3   /   3   
Blank lines     2   /   2   
                    
***Quality***                   
Long functions      3   /   3   
Multipurpose functions      4   /   4   
Unncessessary Code      3   /   3   
                    
Total       100    /  100 
*/

/*==============================================================================
 Author:       =        Tim Siwula
 Liscense:     =        GPLv2
 File:         =        mpi_dijkstra.c
 Version:      =        0.02
 Created:      =        10/28/2015
 ==============================================================================
 Compile:      =        mpicc -g -Wall -o mpi_dijkstra mpi_dijkstra.c
 Run:          =        mpiexec -n <p> ./mpi_dijkstra (on lab machines)
               =        csmpiexec -n <p> ./mpi_dijkstra (on the penguin cluster)
 ==============================================================================
 Purpose:      =        Implement I/O functions that will be useful in an
                        an MPI implementation of Dijkstra's algorithm.
                        In particular, the program creates an MPI_Datatype
                        that can be used to implement input and output of
                        a matrix that is distributed by block columns.  It
                        also implements input and output functions that use
                        this datatype.  Finally, it implements a function
                        that prints out a process' submatrix as a string.
                        This makes it more likely that printing the submatrix
                        assigned to one process will be printed without
                        interruption by another process.
 ==============================================================================
 Input:        =        n:  the number of rows and the number of columns
                            in the matrix
                        mat:  the matrix:  note that INFINITY should be
                              input as 1000000
 ==============================================================================
 Output:       =        The submatrix assigned to each process and the
                        complete matrix printed from process 0.  Both
                        print "i" instead of 1000000 for infinity.
 ==============================================================================
 Note:         =        1.  The number of processes, p, should evenly divide n.
                        2.  You should free the MPI_Datatype object created by
                            the program with a call to MPI_Type_free:  see the
                            main function.
                        3.  Example:  Suppose the matrix is

                               0 1 2 3
                               4 0 5 6
                               7 8 0 9
                               8 7 6 0

                        Then if there are two processes, the matrix will be
                        distributed as follows:

                           Proc 0:  0 1    Proc 1:  2 3
                                    4 0             5 6
                                    7 8             0 9
                                    8 7             6 0
  .......................................................................... */

/* ========================================================================== */
/*                          External libaries                                 */
/* ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
/* .......................................................................... */

/* ========================================================================== */
/*                          Constants                                         */
/* ========================================================================== */
#define MAX_STRING 10000
#define INFINITY 1000000
#define MAX_ELEMENT 20
/* .......................................................................... */

/* ========================================================================== */
/*                          Custom function definitions                       */
/* ========================================================================== */
int Read_n(int my_rank, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Print_paths(int pred[], int n, int global_pred[], int my_rank, int loc_n);
void Print_dists(int dist[], int n, int my_rank, int loc_n,
                  int global_dist[]);
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n, int my_rank,
                  int* min_dist_p);
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
  int my_rank);
void Read_matrix(int loc_mat[], int n, int loc_n,
            MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
/* .......................................................................... */

/* ========================================================================== */
/*                          Main( )                                           */
/* ========================================================================== */
int main(int argc, char **argv)
{
   int n, loc_n, p, my_rank;
   int *loc_mat, *loc_dist, *loc_pred, *global_dist, *global_pred;
   MPI_Datatype blk_col_mpi_t;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &p);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   n = Read_n(my_rank, MPI_COMM_WORLD);
   loc_n = n/p;
   loc_mat = malloc(n*loc_n*sizeof(int));
   global_dist = malloc(n*sizeof(int));
   global_pred = malloc(n*sizeof(int));
   loc_dist = malloc(loc_n*sizeof(int));
   loc_pred = malloc(loc_n*sizeof(int));
   blk_col_mpi_t = Build_blk_col_type(n, loc_n);
   Read_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, MPI_COMM_WORLD);
   Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, n, my_rank);
   Print_dists(loc_dist, n, my_rank, loc_n, global_dist);
   Print_paths(loc_pred, n, global_pred, my_rank, loc_n);
   free(loc_mat);
   free(global_dist);
   free(global_pred);
   free(loc_dist);
   free(loc_pred);
   MPI_Type_free(&blk_col_mpi_t);
   MPI_Finalize();
   return 0;
}  /* main */
/* .......................................................................... */

/* ============================================================================
 Function:     =        Dijkstra
 Purpose:      =        Apply Dijkstra's algorithm to the matrix mat for each
               =        process.
 ==============================================================================
 Input arg:    =        1. loc_mat:  adjacency matrix of graph for each process.
               =        2. loc_dist: each processors cheapest edge.
               =        3. loc_pred: each processors previous vertex in path.
               =        4. loc_n:   each processors range of work.
               =        5. n:  the number of vertices.
               =        6. my_rank: each processors rank.
 ==============================================================================
 Output arg:   =        1. dist:  dist[v] = distance 0 to v.
               =        2. pred:  pred[v] = predecessor of v on a shortest path
               =                            0->v.
 ============================================================================ */
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
   int my_rank){
   int i, loc_u, loc_v, *loc_known, new_dist, min_dist_p;
   loc_known = malloc(loc_n*sizeof(int));
   loc_dist[0] = 0; loc_pred[0] = 0; loc_known[0] = 1;
   for (loc_v = 0; loc_v < loc_n; loc_v++){
      loc_dist[loc_v] = loc_mat[0*loc_n + loc_v];
      loc_pred[loc_v] = 0;
      loc_known[loc_v] = 0;
   }
   if (my_rank == 0) loc_known[0] = 1;
   for (i = 1; i < n; i++){
      loc_u = Find_min_dist(loc_dist, loc_known, loc_n, my_rank, &min_dist_p);
      if (loc_u/loc_n == my_rank)
          loc_known[loc_u % loc_n] = 1;
      for (loc_v = 0; loc_v < loc_n; loc_v++){
         if (!loc_known[loc_v]){
            new_dist = min_dist_p + loc_mat[loc_u*loc_n + loc_v];
            if (new_dist < loc_dist[loc_v]){
               loc_dist[loc_v] = new_dist;          // update vertex
               loc_pred[loc_v] = loc_u;
            }}}} /* for i */
   free(loc_known);
}  /* Dijkstra */
/* .......................................................................... */

/* ============================================================================
 Function:     =        Find_min_dist
 Purpose:      =        Find the vertex u with minimum distance to 0
                         (dist[u]) among the vertices whose distance
                         to 0 is not known.
 ==============================================================================
 Input arg:    =        1. loc_dist:  dist[v] = current estimate of distance
                            0->v
               =        2. loc_known: whether the minimum distance 0-> is known
               =        3. loc_n: each processors range of work.
               =        4. min_dist_p: the current minimum distance.
 ==============================================================================
 Output arg:   =        1. loc_u: The vertex u whose distance to 0, dist[u]
                           is a minimum among vertices whose distance
                           to 0 is not known.
 ============================================================================ */
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n, int my_rank,
      int* min_dist_p){
   int loc_v, loc_min_dist = INFINITY+1, global_u;
   int loc_u = -1;  // local vertex number, ranges from 0 to loc_n = n/p
   int my_min[2], glbl_min[2];
   for (loc_v = 0; loc_v < loc_n; loc_v++){
      if (!loc_known[loc_v]){
         if (loc_dist[loc_v] < loc_min_dist){
            loc_u = loc_v;
            loc_min_dist = loc_dist[loc_v];
         }}}
   if (loc_u != -1) {
      global_u = loc_u + my_rank*loc_n;    // converts loc_u to glbl_u
      my_min[0] = loc_dist[loc_u];       // distance from 0 to loc_u
      my_min[1] = global_u;
   } else {
      global_u = -1;    // converts loc_u to glbl_u
      my_min[0] = INFINITY+1;       // distance from 0 to loc_u
      my_min[1] = global_u;
   }
   MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
   loc_u = glbl_min[1]; // global min vertex
   *min_dist_p = glbl_min[0];
   return loc_u;      // returns the global min vertex
}  /* Find_min_dist */
/* .......................................................................... */

/*-------------------------------------------------------------------
 * Function:    Print_dists
 * Purpose:     Print the length of the shortest path from 0 to each
 *              vertex
 * In args:     n:  the number of vertices
 *              dist:  distances from 0 to each vertex v:  dist[v]
 *                 is the length of the shortest path 0->v
 */
void Print_dists(int loc_dist[], int n, int my_rank, int loc_n,
                  int global_dist[]){
   int v;
   MPI_Gather(loc_dist, loc_n, MPI_INT, global_dist, loc_n,
              MPI_INT, 0, MPI_COMM_WORLD);
   if (my_rank == 0){
     printf("The distance from 0 to each vertex is:\n");
     printf("  v    dist 0->v\n");
     printf("----   ---------\n");
     for (v = 1; v < n; v++)
        printf("%3d       %4d\n", v, global_dist[v]);
     printf("\n");
   }
} /* Print_dists */

/*-------------------------------------------------------------------
 * Function:    Print_paths
 * Purpose:     Print the shortest path from 0 to each vertex
 * In args:     n:  the number of vertices
 *              pred:  list of predecessors:  pred[v] = u if
 *                 u precedes v on the shortest path 0->v
 */
void Print_paths(int pred[], int n, int global_pred[], int my_rank, int loc_n ){
   int v, w, count, i;
   int *path = malloc(n*sizeof(int));
   MPI_Gather(pred, loc_n, MPI_INT, global_pred, loc_n,
              MPI_INT, 0, MPI_COMM_WORLD);
      if (my_rank == 0)
      {
           printf("  v     Path 0->v\n");
           printf("----    ---------\n");
           for (v = 1; v < n; v++){
              printf("%3d:    ", v);
              count = 0;
              w = v;
              while (w != 0){
                 path[count] = w;
                 count++;
                 w = global_pred[w];
              }
              printf("0 ");
              for (i = count-1; i >= 0; i--)
                 printf("%d ", path[i]);
              printf("\n");
           }
         }
   free(path);
}  /* Print_paths */

/*---------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Read in the number of rows in the matrix on process 0
 *            and broadcast this value to the other processes
 * In args:   my_rank:  the calling process' rank
 *            comm:  Communicator containing all calling processes
 * Ret val:   n:  the number of rows in the matrix
 */
int Read_n(int my_rank, MPI_Comm comm) {
   int n;
   if (my_rank == 0)
   {
     printf("Enter the number of verticies for your matrix.\n");
     scanf("%d", &n);
     //printf("\n");
   }
   MPI_Bcast(&n, 1, MPI_INT, 0, comm);
     return n;
}  /* Read_n */

/*---------------------------------------------------------------------
 * Function:  Build_blk_col_type
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            loc_n = n/p:  number cols in the block column
 * Ret val:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype Build_blk_col_type(int n, int loc_n) {
   MPI_Aint lb, extent;
   MPI_Datatype block_mpi_t;
   MPI_Datatype first_bc_mpi_t;
   MPI_Datatype blk_col_mpi_t;

   MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
   MPI_Type_get_extent(block_mpi_t, &lb, &extent);

   MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);
   MPI_Type_create_resized(first_bc_mpi_t, lb, extent,
         &blk_col_mpi_t);
   MPI_Type_commit(&blk_col_mpi_t);

   MPI_Type_free(&block_mpi_t);
   MPI_Type_free(&first_bc_mpi_t);

   return blk_col_mpi_t;
}  /* Build_blk_col_type */

/*---------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read in an nxn matrix of ints on process 0, and
 *            distribute it among the processes so that each
 *            process gets a block column with n rows and n/p
 *            columns
 * In args:   n:  the number of rows in the matrix and the submatrices
 *            loc_n = n/p:  the number of columns in the submatrices
 *            blk_col_mpi_t:  the MPI_Datatype used on process 0
 *            my_rank:  the caller's rank in comm
 *            comm:  Communicator consisting of all the processes
 * Out arg:   loc_mat:  the calling process' submatrix (needs to be
 *               allocated by the caller)
 */
void Read_matrix(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
   int* mat = NULL, i, j;

   if (my_rank == 0)
   {
     printf("Please read in matrix into process 0.\n");
      mat = malloc(n*n*sizeof(int));
      for (i = 0; i < n; i++)
         for (j = 0; j < n; j++)
            scanf("%d", &mat[i*n + j]);
   }

   MPI_Scatter(mat, 1, blk_col_mpi_t,
           loc_mat, n*loc_n, MPI_INT, 0, comm);

   if (my_rank == 0) free(mat);
}  /* Read_matrix */
