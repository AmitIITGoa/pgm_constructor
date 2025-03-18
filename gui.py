import streamlit as st
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
    span_set.add(tuple([0] * n))
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
            continue

        # Swap so that pivot is in the pivot_row if needed
        if row_with_one != pivot_row:
            M[[pivot_row, row_with_one]] = M[[row_with_one, pivot_row]]

        # Eliminate 1's in pivot_col below pivot_row
        for r in range(pivot_row + 1, rows):
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

    # If no nonzero basis vectors are found, return an empty array
    if not basis:
        return np.array([])
    G = np.array(basis)
    return G

# ------------------ Streamlit App ------------------

st.title("Linear Extension & Generator Matrix Tool")

# Input controls for number of vectors and vector size
n = st.number_input("Enter how many vectors to input:", min_value=1, step=1, value=5)
m = st.number_input("Enter the size of each vector:", min_value=1, step=1, value=3)

st.markdown("### Input Vectors")
st.markdown("Enter one vector per line. Separate elements with spaces.")
vectors_input = st.text_area("Vectors:", value="0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0")

if st.button("Compute"):
    try:
        # Process the input text and convert to a list of tuples
        lines = [line.strip() for line in vectors_input.strip().split("\n") if line.strip()]
        if len(lines) != n:
            st.error(f"Expected {n} vectors, but got {len(lines)}.")
        else:
            input_vectors = []
            for line in lines:
                r = list(map(int, line.split()))
                if len(r) != m:
                    st.error(f"Invalid data: Each vector must have exactly {m} elements.")
                    st.stop()
                input_vectors.append(tuple(r))
            
            # Check if all vectors are zero vectors
            if all(vec == (0,) * m for vec in input_vectors):
                st.info("All provided vectors are zero. The minimal extension is only the zero vector, and no non-zero generator matrix can be computed.")
                minimal_extension = get_minimal_linear_extension(input_vectors)
                st.subheader("Minimal Linear Extension (All Codewords in the Span)")
                for vec in minimal_extension:
                    st.write(vec)
            else:
                # 1) Get the minimal linear extension
                minimal_extension = get_minimal_linear_extension(input_vectors)
                st.subheader("Minimal Linear Extension (All Codewords in the Span)")
                for vec in minimal_extension:
                    st.write(vec)
                
                # 2) Construct the generator matrix from the minimal extension
                G = get_generator_matrix(minimal_extension)
                if G.size == 0:
                    st.info("No non-zero generator matrix could be computed from the provided vectors.")
                else:
                    st.subheader("Generator Matrix G (Rows = Basis Vectors)")
                    st.write(G)
                    
                    # 3) Verification: Generate codewords from the generator matrix
                    k = G.shape[0]
                    generated_codewords = {}
                    for msg_int in range(2**k):
                        # Convert msg_int into a length-k binary vector
                        msg = [(msg_int >> i) & 1 for i in range(k)]
                        msg = np.array(msg[::-1])  # Reverse for proper ordering
                        codeword = msg.dot(G) % 2
                        generated_codewords[tuple(msg)] = tuple(codeword)
                    
                    st.subheader("Generated Codewords from All Possible Messages")
                    for msg_int in range(2**k):
                        msg = [(msg_int >> i) & 1 for i in range(k)]
                        msg = np.array(msg[::-1])
                        codeword = generated_codewords[tuple(msg)]
                        st.write(f"Message = {tuple(msg)}  -->  Codeword = {codeword}")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
