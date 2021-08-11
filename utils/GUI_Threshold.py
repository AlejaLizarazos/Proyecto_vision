from ipywidgets import interact, interactive
import ipywidgets as widgets
from ipywidgets import Layout
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io, color
import os
import sys

from utils.funciones_utiles import *


##############################################################
# Show Interact

def show_interact_threshold(function, max_v=0, min_v=0, step=0, lineal=True, images=None,mask=None):
    box_layout = Layout(display='flex',
                        flex_flow='column')

    _ = interact(
        analisis_threshold,
        color_space=widgets.ToggleButtons(
            options=['RGB', 'CMY', 'YIQ', 'YUB', 'HSL', 'HSV', 'LAB', 'XYZ', 'HLS'],
            description='Escoge el espacio de color a usar:',
            disabled=False,
            button_style='info',
            tooltips=['modelo aditivo', 'modelo sustractivo',
                      'separa la información de intensidad o luminancia',
                      'ancho de banda reducido para los componentes de crominancia',
                      'fácilmente interpretables y legibles por un humano ...',
                      '... métodos en los que la representación del componente de color no es lineal',
                      'L -> información sobre luminosidad, a* y b* -> información de color',
                      'sensores de color (XYZ)'],
            layout=box_layout,
            style={'description_width': '200px'}
        ),
        channel=widgets.RadioButtons(description="Escoge el canal a visualizar:", options=[1, 2, 3], style={'description_width': 'auto'}, ),

        max_lim=widgets.FloatSlider(value=1,
                              min=min_v,
                              max=max_v,
                              step=step,
                              description='Max val Threshold:' if function != 'T. gamma' else 'a: ',
                              disabled=False,
                              continuous_update=False,
                              orientation='horizontal',
                              readout=True,
                              readout_format='.1f',
                              layout=Layout(display='none') if (not lineal and function != 'T. gamma') else Layout()
                              ),
        min_lim=widgets.FloatSlider(value=1,
                              min=min_v,
                              max=max_v,
                              step=step,
                              description='Min val Threshold:' if function != 'T. gamma' else 'gamma: ',
                              disabled=False,
                              continuous_update=False,
                              orientation='horizontal',
                              readout=True,
                              readout_format='.1f',
                              layout=Layout(display='none') if (not lineal and function != 'T. gamma') else Layout()
                              ),
        f=widgets.Text(
            value=function,
            placeholder='Type something',
            description='String:',
            disabled=False,
            layout=Layout(display='none')
        ),
        lineal=widgets.Text(options=['RGB', 'CMY', 'YIQ', 'YUB', 'HSL', 'HSV', 'LAB', 'XYZ', 'HLS'],
                            value=str(lineal),
                            placeholder='Type something',
                            description='String:',
                            disabled=False,
                            layout=Layout(display='none')
                            ),

        img=widgets.Dropdown(
            options=[str(image) for image in images], value=str(images[0]), description="Escoger imagen:",
            style={'description_width': 'auto'}
        ),

        save=widgets.ToggleButtons(value=None,
                                   options=['Guardar 3 canales', 'Guardar 1 canal'],
                                   description='Cómo quieres guardar la imagen:',
                                   disabled=False,
                                   button_style='warning',
                                   tooltips=[''],
                                   layout=box_layout,
                                   style={'description_width': '200px'}
                                   ),

    );


##############################################################
# Show Interact

def analisis_threshold(color_space, channel, img, max_lim, min_lim, f, lineal,save):
    img = cv2.imread(img)
    img_space = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    img_tittle = color_space
    channel_tittle = color_space[channel - 1]

    color_space, img_space = get_color_space(img, color_space, img_space)

    # if lineal == 'True':
    #     img_space = apply_linear_function(img_space, f, [0, 0, 0])
    #
    # elif lineal == 'False':
    #     img_space = apply_non_linear_function(img_space, f, [a, b])

    img_channel = img_space[:, :, channel - 1]

    img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # print(mask.shape,img_space.shape)
    fig, arreglo_plots = plt.subplots(2, 2, figsize=(15, 8))
    colors = ('r', 'g', 'b')


    arreglo_plots[0,0].set_title('Canal ' + channel_tittle)
    arreglo_plots[0,0].imshow(img_channel, cmap="gray")


    arreglo_plots[0,1].set_title('Histograma Canal ' + channel_tittle)
    img_array = img_channel.ravel()

    arreglo_plots[0,1].hist(img_array, histtype='step', bins=255,
                          range=(0.0, 255.0), density=True, color=colors[channel - 1])

    arreglo_plots[1,0].set_title(img_tittle)
    arreglo_plots[1,0].imshow(img_RGB)

    out=getmask2(img_RGB)
    # out=cv2.bitwise_and( img_space,mask)
    arreglo_plots[1,1].set_title('Canal ' + channel_tittle)
    arreglo_plots[1,1].imshow(out, cmap="gray")

    plt.show()

    if save == 'Guardar 3 canales':
        cv2.imwrite(os.path.join("results/", img_tittle + '.jpg'), img_space)
    elif save == 'Guardar 1 canal':
        cv2.imwrite(os.path.join("results/", channel_tittle + '.jpg'), img_channel)


def getmask2(img_rgb_):
    mask = cv2.imread('img\panel_mask.jpg')
    img_R_RGB = img_rgb_[:, :, 0]
    img_G_RGB = img_rgb_[:, :, 1]
    img_B_RGB = img_rgb_[:, :, 2]

    img_2 = 0.596 * img_R_RGB - 0.274 * img_G_RGB - 0.322 * img_B_RGB
    # img_2 =cv2.bitwise_and(img_2, )

    img_out = img_rgb_.copy()
    img_out[(img_2 > 2)&(mask[:,:,0]==255)] = [200, 200, 0]
    return img_out