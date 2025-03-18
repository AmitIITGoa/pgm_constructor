import numpy as np
from itertools import combinations

def get_minimal_linear_extension(vectors):
    """
    Compute the minimal linear extension (all linear combinations)
    for a set of binary vectors over GF(2).
    """
    n = len(vectors[0])         
    span_set = set(vectors)  # Start with the original vectors
    
    # Generate all possible linear combinations using XOR (GF(2) addition)
    for r in range(1, len(vectors) + 1):
        for subset in combinations(vectors, r):
            xor_result = np.bitwise_xor.reduce(np.array(subset))
            span_set.add(tuple(xor_result))

    # Ensure the zero vector is included
    span_set.add(tuple([0]*n))

    return sorted(span_set)

def gaussian_elimination_gf2(M):
    """
    Perform Gaussian Elimination over GF(2) on matrix M
    and return the row-echelon form along with the rank.
    """
    M = M.copy()
    rows, cols = M.shape
    pivot_row = 0

    for pivot_col in range(cols):
        # Find a row with a 1 in the pivot_col at or below pivot_row
        row_with_one = None
        for r in range(pivot_row, rows):
            if M[r, pivot_col] == 1:
                row_with_one = r
                break
        if row_with_one is None:
            # No pivot in this column
            continue

        # Swap so that pivot is in the pivot_row if needed
        if row_with_one != pivot_row:
            M[[pivot_row, row_with_one]] = M[[row_with_one, pivot_row]]

        # Eliminate 1's in pivot_col below pivot_row
        for r in range(pivot_row+1, rows):
            if M[r, pivot_col] == 1:
                M[r] = (M[r] + M[pivot_row]) % 2

        pivot_row += 1
        if pivot_row == rows:
            break

    return M, pivot_row  # pivot_row is the rank

def get_generator_matrix(vectors):
    """
    From the input set of vectors, find a basis using Gaussian elimination,
    and return it as the generator matrix G (each row is a basis vector).
    """
    # Convert to numpy array, remove duplicates
    arr = np.unique(np.array(vectors, dtype=int), axis=0)

    # Apply Gaussian elimination
    M, rank = gaussian_elimination_gf2(arr)

    # Re-run an elimination pass to identify the pivot rows clearly
    M_reduced, _ = gaussian_elimination_gf2(M)

    # Collect candidate rows that are nonzero
    candidate_rows = [row for row in M_reduced if np.any(row)]

    # Now we finalize them into a clean basis
    basis = []
    for row in candidate_rows:
        r = row.copy()
        # Reduce this row against previously chosen basis vectors
        for b in basis:
            pivot_pos = np.where(b == 1)[0][0]  # leftmost '1' in b
            if r[pivot_pos] == 1:
                r = (r + b) % 2
        if np.any(r):
            basis.append(r)

    G = np.array(basis)
    return G

# ------------------ Example Usage ------------------

if __name__ == "__main__":

    # Example input vectors
    
    n = int(input("Enter how many vectors to enter: "))
    m = int(input("Enter the size of each vector: "))

    input_vectors = [] 
    for i in range(n):
        r = list(map(int, input("Enter vector elements separated by space: ").split()))
        if len(r) != m:
            print("invalid data")
            import sys
            sys.exit(0)
        input_vectors.append(tuple(r))

    # 1) Get the minimal linear extension:
    minimal_extension = get_minimal_linear_extension(input_vectors)
    print("Minimal Linear Extension (All Codewords in the Span):")
    for vec in minimal_extension:
        print("   ", vec)

    # 2) Construct the generator matrix from the minimal extension
    G = get_generator_matrix(minimal_extension)
    print("\nGenerator Matrix G (rows = basis vectors):")
    print(G)

    # 3) Verification step:
    #    Generate all codewords by all possible messages and compare
    #    to the minimal_extension set.

    # Number of basis vectors = dimension
    k = G.shape[0]

    # We'll enumerate all messages in {0,1}^k
    # and compute codeword = m * G  (in GF(2)).
    generated_codewords = {}
    for msg_int in range(2**k):
        # Convert msg_int into a length-k binary vector
        # e.g. if k=3, then we want 000, 001, 010, 011, etc.
        msg = [(msg_int >> i) & 1 for i in range(k)]
        msg = np.array(msg[::-1])  # reverse so msg[0] is the leftmost bit if desired

        codeword = msg.dot(G) % 2
        generated_codewords[tuple(msg)] = tuple(codeword)

    # Convert minimal_extension to a set for easy comparison
    minimal_extension_set = set(minimal_extension)
    # And collect the generated codewords
    generated_codewords_set = set(generated_codewords.values())

    print("\nGenerated Codewords from All Possible Messages:")
    # We’ll also show which message led to each codeword.
    # Sort by message integer, just for nice ordering:
    for msg_int in range(2**k):
        # Reconstruct the same msg used above
        msg = [(msg_int >> i) & 1 for i in range(k)]
        msg = np.array(msg[::-1])
        codeword = generated_codewords[tuple(msg)]
        print(f"   Message = {tuple(msg)} --> Codeword = {codeword}")

    # 4) Check if they match exactly the minimal extension
    match = (generated_codewords_set == minimal_extension_set)
    print("\nDo the generated codewords match the minimal extension exactly?")
    print("Answer:", match)
