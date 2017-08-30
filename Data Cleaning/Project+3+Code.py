
# coding: utf-8

# In[1]:

#code for creating sample
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow

OSM_FILE = "washington.osm"  # Replace this with your osm file
SAMPLE_FILE = "sample.osm"

k = 20 # Parameter: take every k-th top level element

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag

    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


with open(SAMPLE_FILE, 'wb') as output:
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n'.encode())
    output.write('<osm>\n'.encode())

    # Write every kth top level element
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8'))

    output.write('</osm>'.encode())


# In[9]:

#code for auditing street names

#!/usr/bin/env python
# -*- coding: utf-8 -*

from collections import defaultdict
import re

osm_file = open("sample.osm", "r", encoding='utf8')

street_type_re = re.compile(r'\S+\.?$', re.IGNORECASE)
street_types = defaultdict(int)

def audit_street_type(street_types, street_name): #for each street type it will check against previous ones
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()

        street_types[street_type] += 1

def print_sorted_dict(d):   #function that prints and matches the street endings with their count
    keys = d.keys()
    keys = sorted(keys, key=lambda s: s.lower())
    for k in keys:
        v = d[k]
        print(k, v)

def is_street_name(elem):  #pulls street name from v attribute where k is "addr:street"
    return (elem.tag == "tag") and (elem.attrib['k'] == "addr:street")

def audit():  #function where audit takes place
    for event, elem in ET.iterparse(osm_file):
        if is_street_name(elem):
            audit_street_type(street_types, elem.attrib['v'])    
    print_sorted_dict(street_types)    

if __name__ == '__main__':
    audit()


# In[10]:

#cleaning of street names and quadrants
import pprint
import re
from collections import defaultdict
import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow

OSMFILE = open("sample.osm", "r", encoding='utf8')

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

#expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
#            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping_street = { "St": "Street",
            "St.": "Street",
            "Rd": "Road",
            "Rd.": "Road",
            "Ave": "Avenue",
            "Ave.": "Avenue",
            "Blvd": "Boulevard",
            "Blvd.": "Boulevard",
            "Dr": "Drive",
            "Dr.": "Drive",
            "Ct": "Court",
            "Ct.": "Court",
            "Pl": "Place",
            "Pl.": "Place",
            "Sq": "Square",
            "Sq.": "Square",
            "Ln": "Lane",
            "Ln.": "Lane",
            "PWY": "Parkway",
            "PWY.": "Parkway",
            "CMNS": "Commons",
            "CMNS.": "Commons",
            "Tr": "Trail",
            "Tr.": "Trail",
            "Northwest": "Northwest",
            "NW": "Northwest",
            "Northeast":"Northeast",
            "NE":"Northeast",
            "Southeast":"Southeast",
            "SE":"Southeast",
            "Southwest":"Southwest",
            "SW":"Southwest",
          }

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        #if street_type not in expected:
        street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):  #creates a list of all the streets to update, not
    osm_file = OSMFILE
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types

def update_name(name, mapping_street):   #this code actually executes the name updates
    m = street_type_re.search(name)   
    if m:       
        street_type = m.group()      
    if street_type in mapping_street:         
        name = re.sub(street_type_re, mapping_street[street_type], name)   
    return name

def test():   #executes the cleaning and will print any changeg it made.
    st_types = audit(OSMFILE)
  #  assert len(st_types) == 3
  #  pprint.pprint(dict(st_types))

    for name, ways in st_types.items():
            better_name = update_name(name, mapping_street)
            if name != better_name:
                print (name, "=>", better_name)

if __name__ == '__main__':
    test()


# In[8]:

#auditing zipcodes, no changes needed
osm_file = open("sample.osm", "r", encoding='utf8')

post_type_re = re.compile(r'\S+\.?$', re.IGNORECASE)
postcodes = defaultdict(int)

def audit_post_type(postcodes, postcode): #checks zipcodes and will add it to its respective count
    m = post_type_re.search(postcode)
    if m:
        postcode = m.group()

        postcodes[postcode] += 1

def print_sorted_dict(d):  #prints dictionary of all zipcodes and their frequency count
    keys = d.keys()
    keys = sorted(keys, key=lambda s: s.lower())
    for k in keys:
        v = d[k]
        print(k, v)

def is_postcode(elem): #finds k attribute "postcode"
    return (elem.tag == "tag") and (elem.attrib['k'] == "addr:postcode")

def audit():   #where audit takes place
    for event, elem in ET.iterparse(osm_file):
        if is_postcode(elem):
            audit_post_type(postcodes, elem.attrib['v'])    
    print_sorted_dict(postcodes)  #uncomment to print all zips
    osm_file.close()

if __name__ == '__main__':
    audit()


# In[11]:

#fixing cuisine capitalization error
import pprint
import re
from collections import defaultdict
import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow

OSMFILE = open("sample.osm", "r", encoding='utf8')

cuisine_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

# UPDATE THIS VARIABLE
mapping = {"Afghan" : "afghan",
            "burger;american" : "burger",
            "cajun;creole":"cajun",
            "coffee":"coffee_shop",
            "Diner":"diner",
            "Ethiopian":"ethiopian",
            "Jamaican":"jamaican",
            "Korean":"korean",
            "latin_america":"latin_american",
            "Salad":"salad",
            "salads":"salad",
            "steak":"steak_house"
          }

def audit_cuisine_type(cuisine_types, cuisine_name): #creates dict of cuisine types
    m = cuisine_type_re.search(cuisine_name)
    if m:
        cuisine_type = m.group()
        cuisine_types[cuisine_type].add(cuisine_name)


def is_cuisine_name(elem):  # return type for cuisine types
    return (elem.attrib['k'] == "cuisine")


def audit(osmfile):
    osm_file = OSMFILE
    cuisine_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_cuisine_name(tag):
                    audit_cuisine_type(cuisine_types, tag.attrib['v'])
    osm_file.close()
    return cuisine_types

def update_name(name, mapping): #where names are changed according to its mapping  
    m = cuisine_type_re.search(name)   
    if m:       
        cuisine_type = m.group()      
    if cuisine_type in mapping:         
        name = re.sub(cuisine_type_re, mapping[cuisine_type], name)   
    return name

def test(): #update_name function is called and a printout of all changes occurs
    st_types = audit(OSMFILE)
  #  assert len(st_types) == 3
  #  pprint.pprint(dict(st_types))

    for name, ways in st_types.items():
            better_name = update_name(name, mapping)
            if name != better_name:
                print (name, "=>", better_name) 

if __name__ == '__main__':
    test()


# In[6]:

import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET

import cerberus

import schema

OSM_PATH = "sample.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements

# Node ---------------------------
    if element.tag == 'node':
        for attribute in node_attr_fields: #pop kv's for nodes
                node_attribs[attribute] = element.attrib[attribute]
        for secondary_elem in element.findall('tag'): #pop secondary node tags
            # secondary_elem = clean_data(secondary_elem)
            if secondary_elem.attrib['k'] == "addr:street" :
                name = secondary_elem.attrib['v']
                m = street_type_re.search(name)   
                if m:       
                    street_type = m.group()      
                if street_type in mapping_street:         
                    name = re.sub(street_type_re, mapping_street[street_type], name)
                    secondary_elem.attrib['v'] = name
            elif secondary_elem.attrib['k'] == "cuisine" :
                name = secondary_elem.attrib['v']
                w = cuisine_type_re.search(name)   
                if w:       
                    cuisine_type = w.group()      
                if cuisine_type in mapping:         
                    name = re.sub(cuisine_type_re, mapping[cuisine_type], name) 
                    secondary_elem.attrib['v'] = name
            tag_append = find_tags(secondary_elem, element.attrib['id'])
            if tag_append:  #if tag_append is none it will skip the line (if prob char occurs)
                tags.append(tag_append)
        return {'node': node_attribs, 'node_tags': tags}
 #Way---------------------------
    elif element.tag == 'way':
        for attribute in way_attr_fields:
            way_attribs[attribute] = element.attrib[attribute]
        for secondary_elem in element.findall('tag'):
            # secondary_elem = clean_data(secondary_elem)
            if secondary_elem.attrib['k'] == "addr:street" :
                name = secondary_elem.attrib['v']
                m = street_type_re.search(name)   
                if m:       
                    street_type = m.group()      
                if street_type in mapping_street:         
                    name = re.sub(street_type_re, mapping_street[street_type], name)
                    secondary_elem.attrib['v'] = name
            elif secondary_elem.attrib['k'] == "cuisine":
                name = secondary_elem.attrib['v']
                w = cuisine_type_re.search(name)   
                if w:       
                    cuisine_type = w.group()      
                if cuisine_type in mapping:         
                    name = re.sub(cuisine_type_re, mapping[cuisine_type], name) 
                    secondary_elem.attrib['v'] = name
            tag_append = find_tags(secondary_elem, element.attrib['id'])
            if tag_append:
                tags.append(tag_append)
        position = 0
        for secondary_elem in element.findall('nd'):
            way_nodes_append = {'id' : element.attrib['id'],
                                'node_id' : secondary_elem.attrib['ref'],
                                'position' : position
                                }
        position != 1
        way_nodes.append(way_nodes_append)
    return{'way': way_attribs, 'way_nodes': way_nodes, 'way_tags' : tags}

def find_tags(elem, id_value):
    key = elem.attrib['k']
    
    if ':' in key:
        key_split = key.split(':')
        type_field = key_split[0]
        key = key[len(type_field)+1:]
    else:
        type_field = 'regular'
    tag_append = {'id' : id_value,
                  'key' : key,
                  'value' : elem.attrib['v'],
                  'type' : type_field
                 }
    
    return tag_append


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))

# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w', encoding='utf8') as nodes_file,          codecs.open(NODE_TAGS_PATH, 'w', encoding='utf8') as nodes_tags_file,          codecs.open(WAYS_PATH, 'w', encoding='utf8') as ways_file,          codecs.open(WAY_NODES_PATH, 'w', encoding='utf8') as way_nodes_file,          codecs.open(WAY_TAGS_PATH, 'w', encoding='utf8') as way_tags_file:

        nodes_writer = csv.DictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = csv.DictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = csv.DictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = csv.DictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = csv.DictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    # Note: Validation is ~ 10X slower. For the project consider using a small
    # sample of the map when validating.
     process_map(OSM_PATH, validate=True)


# In[ ]:




# In[ ]:




# In[ ]:



