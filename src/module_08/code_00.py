import os, json

# define a relative path to the coco annotations JSON definitions
json_annotations_file_path = os.path.join('data','coco','337975.json')

# slurp up the data file
with open(json_annotations_file_path) as fdat: data = json.load(fdat)

# blab about the data type
print('imported json annotations as type: {}'.format(type(data)))

# print the number of annotations listed
print('number of annotations imported: {}'.format(len(data['annotations'])))

# iterate over annotations and print out category
for annotation in data['annotations']:
    print('annotation id: {}, category of object: {}, with bounding box: {}'.format(
        annotation['id'         ],
        annotation['category_id'],
        annotation['bbox'       ]
        )
    )

'''
    Simple import and manipulation of MS COCO annotation data for particular image
    We need the full COCO annotations file to decode the category ID's to something 
    humany readable (e.g. 'human', 'car','dog').  The bounding box format for COCO
    data is ymin,xmin,ymax,xmax in pixels. 
    
    Notice, that each image has a different number of annotations. Therefore, the 
    TFRecord that we create will each have a different number of bboxes.  To do 
    this variable length encodeing we will need to use tf.VarLenFeature() so that
    each record can have appropriate number of annotations.  
'''