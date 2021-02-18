#
# Configuration
#

IN_BITS = "../../data/bit_vector_test50_padded256.npy"
LIMIT = 50

#IN_BITS = "../../data/bit_vector_train50_padded256.npy"
#LIMIT = 100000

DEBUG = False

#
# 
#

import numpy
import os
import sys

if not os.path.exists(IN_BITS):
	print("file %s does not exist." % IN_BITS)
	sys.exit(1)

fname = os.path.basename(IN_BITS)
name, ext = fname.split(".")

in_arr = numpy.load(IN_BITS)
print("input", in_arr.shape, in_arr.dtype)

no_arrs = in_arr.shape[0]
ilen = in_arr.shape[1]
olen = in_arr.shape[1]*8
if LIMIT<no_arrs:
	no_arrs = LIMIT

opath = os.path.join( os.path.dirname(IN_BITS), "%s_f32_%d.%s" % ( name,  no_arrs, ext ) )
print(opath)
if os.path.exists(opath):
	print("possible output file already exists %s" % opath)
	sys.exit(1)
	

out_arr = numpy.zeros( (no_arrs, olen ), numpy.float32 )
if DEBUG: print("output", out_arr.shape, out_arr.dtype)

for i in range(no_arrs):

	if ( i%1000==0): print("processing %d/%d" % (i+1, no_arrs))		

	# iterate over number of arrays
	
	for j in range(ilen):

		# iterate over number of int8's ( elements in input array )

		oval = in_arr[i,j]
		if DEBUG: print("oval(%d,%d): %x %s" % (i,j,oval,type(oval)))

		for k in range(8):

			# iterate over bits in int8

			fval = (oval<<k)&(0x80)
			bval = (fval>0)*1.0
			if DEBUG: print("%d: %x" % (k,bval>0))

			# set the float into the out array
			out_arr[i, j*8 + k ] = bval

	if DEBUG: 
		out_lst = out_arr[i].tolist()
		print("out(%d)" % (i), out_lst)

	if i==(LIMIT-1):
		print("early exist because %d < %d" % (LIMIT, no_arrs), out_arr.shape)
		break

	#break

print("Saving array at %s" % opath)

numpy.save( opath,  out_arr)

print("Done.")
