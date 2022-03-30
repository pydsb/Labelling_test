import os 
from pathlib import Path
import sys

import random
import pandas as pd
import numpy as np
from spiepy import NANOMAP
import Nanonis_functions as NF

from bokeh.layouts import column, gridplot, row
from bokeh.colors import RGB
from bokeh.models import Button, ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.palettes import Greys256
from bokeh.plotting import figure, curdoc

Input_directory = Path('./Small_test')
Output_directory = Path('./Output_labels.d')

Output_filename = 'bokeh_testing.csv'
Output_path = Output_directory.joinpath(Output_filename)
show_table = True

# Initialise the data lists
labels = []
File_names = []
image_num = 0

Filenames = random.sample(os.listdir(Input_directory), len(os.listdir(Input_directory)))

end_im = len(Filenames) - 1

file = Input_directory.joinpath(Filenames[image_num])
if file.suffix == '.sxm':

    _, _, img = NF.get_image_data(file)
    img, _ = NF.flatten_by_line(img)
    img = img - img.mean()

NANOMAP_rgb = (255 * NANOMAP(range(256))).astype('int')
NANOMAP_palette = [RGB(*tuple(rgb)).to_hex() for rgb in NANOMAP_rgb]
# create a plot and style its properties
p = figure(title=f'{Filenames[image_num]} \nImage number {image_num}')
p.image(image=[img], x=0, y=0, dw=2, dh=2, palette=NANOMAP_palette)

if show_table:
    source = ColumnDataSource(dict(files=File_names, labels=labels))
    columns = [
            TableColumn(field="files", title="Filename"),
            TableColumn(field="labels", title="Label")
        ]
    table = DataTable(source=source, columns=columns, width=400, height=280)

p.outline_line_color = None
p.axis.visible = False
p.grid.grid_line_color = None
p.title.text = f'{Filenames[image_num]}\nImage Number {image_num}'
p.title.align = "center"
p.title.text_color = "black"
p.title.text_font_size = "25px"

# create a callback that adds a number in a random location
def Next_im():
    global image_num
    global end_im

    image_num += 1
    if image_num <= end_im:
        file = Input_directory.joinpath(Filenames[image_num])
        if file.suffix == '.sxm':

            _, _, img = NF.get_image_data(file)
            img, _ = NF.flatten_by_line(img)
            img = img - img.mean()
        
        p.title.text = f'{Filenames[image_num]}\nImage Number {image_num}'
        p.image(image=[img], x=0, y=0, dw=2, dh=2, palette=NANOMAP_palette)

        if show_table:
            source.data['files'] = File_names
            source.data['labels'] = labels
    elif image_num == end_im + 1:

        p.title.text = 'Finished!!'                                                     
        p.image(image=[np.zeros((1000,1000))], x=0, y=0, dw=2, dh=2, palette= Greys256)     #No idea why this isn't working.. 
        # p.image.visible = False
        if show_table:
            source.data['files'] = File_names
            source.data['labels'] = labels

        # Set up and save pandas DataFrame
        Labels_data = pd.DataFrame({'Filenames': File_names, 'Labels': labels })

        Labels_data.to_csv(Output_path)
        print('This is now finished')  
        sys.exit()      
    
def Good():
    File_names.append(Filenames[image_num])
    labels.append('Good')
    # print(File_names, labels)

def Medium():
    File_names.append(Filenames[image_num])
    labels.append('Medium')
    # print(File_names, labels)

def Bad():
    File_names.append(Filenames[image_num])
    labels.append('Bad')
    # print(File_names, labels)

def Double():
    File_names.append(Filenames[image_num])
    labels.append('Double')
    # print(File_names, labels)

def Tip_Change():
    File_names.append(Filenames[image_num])
    labels.append('Tip Change')
    # print(File_names, labels)

# add a button widget and configure with the call back
Good_button = Button(label='Good', button_type = 'success' )
Med_button = Button(label='Medium', button_type = 'warning')
Bad_button = Button(label='Bad', button_type = 'danger')
Double_button = Button(label='Double', button_type = 'primary')
tc_button = Button(label='Tip Change', button_type = 'default')

Good_button.on_click(Good)
Med_button.on_click(Medium)
Bad_button.on_click(Bad)
Double_button.on_click(Double)
tc_button.on_click(Tip_Change)

Good_button.on_click(Next_im)
Med_button.on_click(Next_im)
Bad_button.on_click(Next_im)
Double_button.on_click(Next_im)
tc_button.on_click(Next_im)

# put the button and plot in a layout and add to the document

plot = gridplot([Good_button ,Med_button ,Bad_button ,Double_button ,tc_button , p], ncols=1)
# curdoc().add_root(column(Good_button ,Med_button ,Bad_button ,Double_button ,tc_button , p))
# curdoc().add_root(p)

if show_table:
    curdoc().add_root(row(plot, table))
else:
    curdoc().add_root(plot)

# os.popen('bokeh serve --show Playing_around.py')