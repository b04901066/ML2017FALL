import os, sys, numpy, skimage.io
img_dim = (600, 600, 3)

def out_trans(M):
    M -= numpy.min(M)
    M /= numpy.max(M)
    return (M * 255).astype(numpy.uint8)

img_paths = os.listdir( sys.argv[1] )
X = numpy.zeros((numpy.prod(img_dim), len(img_paths) ), dtype=numpy.float)
for i in range(len(img_paths)):
    X[:,i] = skimage.io.imread(os.path.join(sys.argv[1], img_paths[i])).reshape(-1)
print('X(M, N):', X.shape)
X_mean = numpy.average(X, axis=1).reshape(-1, 1)

U, s, V = numpy.linalg.svd( X - X_mean , full_matrices=False)
k = 4
weights = numpy.dot( (skimage.io.imread( os.path.join( sys.argv[1], sys.argv[2] ) ).flatten() - X_mean.reshape(1, -1)), numpy.copy( U[ : , :k ]))
weights = weights.reshape(4, 1)
A3output = numpy.dot( U[ : , :k ] , weights )
A3output += X_mean
skimage.io.imsave( 'reconstruction.jpg', out_trans( A3output.reshape( img_dim ) ) )