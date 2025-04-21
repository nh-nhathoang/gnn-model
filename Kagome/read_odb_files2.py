import glob
import odbAccess

# Width of the domain (to calculate elastic modulus)
W = 100.

# Elastic modulus of parent material (to calculate E/Es)
Es = 200000.

# Create file to output results and write header
output_file = open('Results.txt', 'w')
output_file.write('Filename, E/Es\n')

# List of all odb files in current folder
files = glob.glob('*.odb')

# Print the files
for file in files:
    # Open odb file
    Odb_file = odbAccess.openOdb(path=file, readInternalSets=True)
    
    # for the master node in History output
    HR_list = Odb_file.steps['Step-1'].historyRegions.keys()
    
    # Total force applied to the lattice
    RF = Odb_file.steps['Step-1'].historyRegions[HR_list[0]].historyOutputs['RF2'].data[-1][1]
    
    # Write results to output file
    output_file.write('{0}, {1}\n'.format(file,RF/W/0.01/Es)) 

    # Close odb file
    Odb_file.close()
    
# Close output file
output_file.close()    