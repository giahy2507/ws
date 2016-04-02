__author__ = 'HyNguyen'

from xml.etree.ElementTree import iterparse
import copy

class ItemXML (object):
    def __init__(self, dict_item):
        self.content = dict_item['CONTENT']
        self.label = dict_item['LABEL']
        self.sentiment_vector_score = None

    @classmethod
    def item_xml_from_dictionany(cls, dict_item):
        return ItemXML(dict_item)

def LoadResultXML(file):
    doc = iterparse(file, ('start', 'end'))
    # Skip the root element
    next(doc)

    items = []
    content = []
    label = -1

    tag_stack = []
    elem_stack = []

    item = None

    count = 0

    for event, elem in doc:
        if event == 'start':
            if elem.tag == 'ITEM':
                item = {}
                label = -1
                content.clear()
            tag_stack.append(elem.tag)
            elem_stack.append(elem)
        elif event == 'end':
            if elem.tag == 'ITEM':
                item['CONTENT'] = copy.deepcopy(content)
                item['LABEL'] = label
                items.append( ItemXML(item) )
                count+=1
                if count % 10000 == 0:
                    print("finished :" , count)
            elif elem.tag == 'SENTENCE':
                content.append(elem.text)
            elif elem.tag == 'LABEL':
                label = int(elem.text)
            try:
                tag_stack.pop()
                elem_stack.pop()
            except IndexError:
                pass

    return items

if __name__ == '__main__':

    items = LoadResultXML('/home/hynguyen/Downloads/result_126k_0.xml')
    print('Length of items = ', len(items))