# Leibniz Harmonic Triangle 

# Print Leibniz Harmonic 
# Triangle 
def LeibnizHarmonicTriangle(n): 
	C = [[0 for x in range(n + 1)] 
			for y in range(n + 1)]; 
			
	# Calculate value of Binomial 
	# Coefficient in bottom up manner 
	for i in range(0, n + 1): 
		for j in range(0, min(i, n) + 1): 
			
			# Base Cases 
			if (j == 0 or j == i): 
				C[i][j] = 1; 
				
			# Calculate value using 
			# previously stored values 
			else: 
				C[i][j] = (C[i - 1][j - 1] +
						C[i - 1][j]); 
							
	# printing Leibniz 
	# Harmonic Triangle 
	for i in range(1, n + 1): 
		for j in range(1, i + 1): 
			print("1/", end = ""); 
			print(i * C[i - 1][j - 1], 
						end = " "); 
		print(); 

LeibnizHarmonicTriangle(4); 
