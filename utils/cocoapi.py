import json
import time

from collections import defaultdict


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class CocoPanoptic:
    def __init__(self, panoptic_file=None):
        """load data set"""
        self.dataset, self.imgs, self.imgToAnn = dict(), dict(), dict()
        if not panoptic_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(panoptic_file, 'r'))
            if not type(dataset) == dict:
                print('annotation file format {} not supported'.format(type(dataset)))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        """create index"""
        print('creating index...')
        imgs, imgToAnn = {}, {}

        for img in self.dataset['images']:
            imgs[img['id']] = img

        for ann in self.dataset['annotations']:
            imgToAnn[ann['image_id']] = ann

        self.imgs = imgs
        self.imgToAnn = imgToAnn

        print('index created!')

    def getImgIds(self):
        """get all image_ids"""
        return list(self.imgs.keys())

    def loadAnns(self, imgIds=[]):
        """load panoptic annotations by image_ids"""
        if _isArrayLike(imgIds):
            return [self.imgToAnn[imgId] for imgId in imgIds]
        elif type(imgIds) == int:
            return [self.imgToAnn[imgIds]]

    def loadImgs(self, ids):
        """load image infos by image_ids"""
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]


class CocoCaption():
    def __init__(self, caption_file=None):
        """load data set"""
        self.imgToAnns = defaultdict(list)
        if not caption_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(caption_file, 'r'))
            if not type(dataset) == dict:
                print('annotation file format {} not supported'.format(
                    type(dataset)))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        """create index"""
        print('creating index...')
        imgToAnns = defaultdict(list)
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
        self.imgToAnns = imgToAnns
        print('index created!')

    def loadAnns(self, imgIds):
        """load captions by image_ids"""
        if _isArrayLike(imgIds):
            return [self.imgToAnns[imgId] for imgId in imgIds]
        elif type(imgIds) == int:
            return [self.imgToAnns[imgIds]]
