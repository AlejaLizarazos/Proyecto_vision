def rgb2hsl(img_regb):
    #Para la conversión RGB - HSL debe garantizarse que las matrices coincidan en tamaño
    #y tipo de datos
    tam = np.shape(img_rgb)
    img_hsl =np.zeros((tam), dtype=np.float32)

    for i in range(tam[0]):
        for j in range(tam[1]):
            #sacar el máximo y el mínimo valor al recorrer las filas y columnas
            max_val = np.max(img_rgb[i][j])
            min_val = np.min(img_rgb[i][j])
            #crear los canales S y L del espacio HSL mediante transformaciones lineales
            s = max_val - min_val
            l = s/2
            #asignación de los canales a la matriz img_hsl
            img_hsl[i][j][1] = s
            img_hsl[i][j][2] = l
            #asignación de valores al canal H
            if(max_val==min_val):
                img_hsl[i][j][0] = 0
                continue
            #extracción de los canales del espacio RGB de la imagen
            red = img_rgb[i][j][0]
            green = img_rgb[i][j][1]
            blue = img_rgb[i][j][2]
            #normalización de los datos, en caso de que el vector max_val sea exactamente 
            #igual a uno de los canales del espacio RGB
            if(max_val == red):
                h = (green-blue)*60/(max_val-min_val)
            elif(max_val == green):
                h = (blue-red)*60/(max_val-min_val) + 120
            else:
                h = (red-green)*60/(max_val-min_val) + 240
            #condicional para que cada valor de h esté acotado entre 0 y 360 en cada iteración
            if h >= 0:
                img_hsl[i,j,0]=h
            else:
                img_hsl[i,j,0] = 360.0 - h
                
    return img_hsl

def apply_linear_function(img, f, args):
    
    import numpy as np
    import cv2
    #Crear una matriz de ceros del tamaño de la imagen de entrada
    res = np.zeros(img.shape, np.uint8)
    #Aplicar la transformación f sobre cada canal del espacio de color
    
    if f == 'suma':
        res[:,:,0] = cv2.add(img[:,:,0], args[0])
        res[:,:,1] = cv2.add(img[:,:,1], args[1])
        res[:,:,2] = cv2.add(img[:,:,2], args[2])
        
    elif f == 'resta':
        res[:,:,0] = cv2.subtract(img[:,:,0], args[0])
        res[:,:,1] = cv2.subtract(img[:,:,1], args[1])
        res[:,:,2] = cv2.subtract(img[:,:,2], args[2])
        
    else:
        
        res[:,:,0] = cv2.multiply(img[:,:,0], args[0])
        res[:,:,1] = cv2.multiply(img[:,:,1], args[1])
        res[:,:,2] = cv2.multiply(img[:,:,2], args[2])
        
    return res


#Definir la función ecualización del histograma cuyo parámetro es una imagen
def histogram_equalization(img):
     
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    #Crear matriz de ceros del tamaño de la imagen y tipo de dato flotante
    res = np.zeros(img.shape, np.float32)
    #Crear un vector 1-D de la matriz de la imagen, es decir "aplanarla"
    img_raveled = img.ravel()
    #Generar el histograma normalizado de la imagen 
    hist_norm = plt.hist(img_raveled, bins=255, range=(0.0, 255.0), density=True)
    #Limpiar la figura actual
    plt.clf()
    #hist_norm[0] es un vector de probabilidades. Añadir al vector, el valor [1 - sumatoria de sus datos]
    pdf = hist_norm[0]
    np.append(pdf, 1.0 - np.sum(pdf))
    #Dado que se añade un dato, se realiza la sumatoria de valores sobre todo el vector
    cdf = [np.sum(pdf[0:x]) for x in range(0,256)]
    #Hallar el valor mínimo y máximo de la imagen
    gmin = np.min(img)
    gmax = np.max(img)
    
    #Generar F(g) - Función de ecualización
    for g in range(0,256):
        res[img == g] = (gmax - gmin)*cdf[g] + gmin
    #Asegurar que los datos sean uint8 y esten en el rango correspondiente
    res[res<0] = 0
    res[res>255] = 255
    res = res.astype(np.uint8)
    return res


def histogram_expansion(img):
    
    import numpy as np
    import cv2
    #Crear matriz de ceros del tamaño de la imagen y tipo de dato flotante
    res = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    
    #Extraer el mínimo y el máximo del conjunto de datos
    m = float(np.min(img))
    M = float(np.max(img))
    #Aplicar la función de expansión(normalización) y asegurar datos uint8
    res = (img-m)*255.0/(M-m)
    res = res.astype(np.uint8)
    
    return res

def gamma_correction(img, a, gamma):
    
    import numpy as np
    import cv2
    #Crear copia de la imagen tipo flotante dada la normalización
    img_copy = img.copy().astype(np.float32)/255.0
    #La función corrección gamma es de la forma ax^gamma, donde x es la imagen de entrada
    res_gamma = cv2.pow(img_copy,gamma)
    res = cv2.multiply(res_gamma, a)
    
    #Asegurar que la los datos queden entre 0 y 255 y sean uint8
    res[res<0] = 0
    res = res*255.0
    res[res>255] = 255
    
    res = res.astype(np.uint8)
    
    return res

def apply_non_linear_function(img, f, args):
    
    import numpy as np
    import cv2
    #Crear una matriz de ceros del tamaño de la imagen de entrada
    res = np.zeros(img.shape, np.uint8)
    
    if f == 'T. gamma':
        res = gamma_correction(img,args[0],args[1])

    elif f == 'Ec. histograma':
        res = histogram_equalization(img)

    else:
        res = histogram_expansion(img)



    
    return res
