import cv2, os
import numpy as np

    
def detect_and_save_faces(name, roi_size):
    
    # define where to look for images and where to save the detected faces
    dir_images = "data/{}".format(name)
    dir_faces = "data/{}/faces".format(name)
    if not os.path.isdir(dir_faces): os.makedirs(dir_faces)  
    
    # put all images in a list
    names_images = [name for name in os.listdir(dir_images) if not name.startswith(".") and name.endswith(".jpg")] # can vary a little bit depending on your operating system
    
    # detect for each image the face and store this in the face directory with the same file name as the original image
    
    ## TODO ##
    face_cascade = cv2.CascadeClassifier(os.path.join("data/",'haarcascade_frontalface_alt.xml'))
    for i in names_images:
        img = cv2.imread(os.path.join(dir_images,i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray)[0]
        #(x,y,w,h) = face
        face_img = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]

        cv2.imwrite(os.path.join(dir_faces,i),cv2.resize(face_img,roi_size))
            
    
def do_pca_and_build_model(name, roi_size, numbers):
    
    # define where to look for the detected faces
    dir_faces = "data/{}/faces".format(name)
    
    # put all faces in a list 
    # #miért van ott a 0_?
    #list not used.
    names_faces = ["0_{}.jpg".format(n) for n in numbers]
    
    # put all faces as data vectors in a N x P data matrix X with N the number of faces and P=roi_size[0]*roi_size[1] the number of pixels
    ## TODO ##
    
    N = np.shape(names_faces)[0]
    P = roi_size[0]*roi_size[1]
    X = np.zeros((N,P))
    for i in numbers:
        img = cv2.imread(os.path.join(dir_faces,"{}.jpg".format(i)),0)
        k = np.reshape(img,(1,P))
        X[i-1,:] = k
        z= np.shape(X)
        
    # calculate the eigenvectors of X
    mean, eigenvalues, eigenvectors = pca(X, number_of_components=P)
    return [mean, eigenvalues, eigenvectors]
    

def test_images(name, roi_size, numbers, models):

    # define where to look for the detected faces
    dir_faces = "data/{}/faces".format(name)
    
    # put all faces in a list
    #0_ minek?
    #names_faces = ["0_{}.jpg".format(n) for n in numbers]
    names_faces = ["{}.jpg".format(n) for n in numbers]
    
    # put all faces as data vectors in a N x P data matrix X with N the number of faces and P=roi_size[0]*roi_size[1] the number of pixels
    ## TODO ##
    N = np.shape(names_faces)[0]
    P = roi_size[0]*roi_size[1]
    X = np.zeros((N,P))
    z = 0
    for i in names_faces:
        img = cv2.imread(os.path.join(dir_faces,i),0)
        k = np.reshape(img,(1,P))
        X[z,:] = k
        z = z+1

        
        
    # reconstruct the images in X with each of the models provided and also calculate the MSE

    # store the results as [[results_model_arnold_reconstructed_X, results_model_arnold_MSE], [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    results = []
    for model in models:
        projections, reconstructions = project_and_reconstruct(X, model)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        results.append([reconstructions, mse])

    return results
    

def pca(X, number_of_components):
    
    ## TODO ##
    mean = np.mean(X,axis = 0)
    #cov = np.dot(np.transpose((X-mean)),(X-mean))/(np.shape(X)[0]-1)
    #cov = (cov+np.transpose(cov))/2
    cov = np.cov(np.transpose(X))
    eigenvalues,eigenvectors = np.linalg.eig(cov)
        
    eigenvectors = np.real(eigenvectors)
    eigenvalues = np.real(eigenvalues)

    pairs = [0]*np.shape(eigenvalues)[0]
    for i in range(np.shape(eigenvalues)[0]):
        pairs[i] =  [np.abs(eigenvalues[i]),eigenvectors[:,i]]

    pairs = sorted(pairs,key = lambda pairs:pairs[0], reverse = True)

    eigenvalues = eigenvalues[:number_of_components]
    eigenvectors = eigenvectors[:number_of_components,:]

    return [mean, eigenvalues, eigenvectors]


def project_and_reconstruct(X, model):
    #model = np.matrix(model)
    cov = np.matrix.transpose(model[2])
    projections = X.dot(cov)
    reconstructions = projections.dot(np.matrix.transpose(cov))+model[0]
    ident = cov*np.matrix.transpose(cov)
    k = 1
    ## TODO ##
    
    return [projections, reconstructions]


if __name__ == '__main__':
    
    roi_size = (30, 30) # reasonably quick computation time
    
    # Detect all faces in all the images in the folder of a person (in this case "arnold" and "barack") and save them in a subfolder "faces" accordingly
    detect_and_save_faces("arnold", roi_size=roi_size)
    detect_and_save_faces("barack", roi_size=roi_size)
    
    # visualize detected ROIs overlayed on the original images and copy paste these figures in a document 
    ## TODO ## # please comment this line when submitting
    
    # Perform PCA on the previously saved ROIs and build a model=[mean, eigenvalues, eigenvectors] for the corresponding person's face making use of a training set
    model_arnold = do_pca_and_build_model("arnold", roi_size=roi_size, numbers=[1, 2, 3, 4, 5, 6])
    model_barack = do_pca_and_build_model("barack", roi_size=roi_size, numbers=[1, 2, 3, 4, 5, 6])
    
    # visualize these "models" in some way (of your choice) and copy paste these figures in a document
    ## todo ## # please comment this line when submitting
    
    # test and reconstruct "unseen" images and check which model best describes it (wrt mse)
    # results=[[results_model_arnold_reconstructed_x, results_model_arnold_mse], [results_model_barack_reconstructed_x, results_model_barack_mse]]
    # the correct model-person combination should give best reconstructed images and therefor the lowest mses
    results_arnold = test_images("arnold", roi_size=roi_size, numbers=[7, 8], models=[model_arnold, model_barack])
    results_barack = test_images("barack", roi_size=roi_size, numbers=[7, 8, 9, 10], models=[model_arnold, model_barack])

    for i in results_arnold[0][0]:
        print('szaragy')
        cv2.imwrite('baszki.jpg',np.reshape(i,roi_size))
    
    # visualize the reconstructed images and copy paste these figures in a document
    ## todo ## # please comment this line when submitting
