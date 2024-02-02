# visualize results
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects

def check_data(dataset, img_tensor, boxlist, mode='annotation', rel_num=30):
    img_tensor = img_tensor.permute(1,2,0).cpu()[:,:,[2,1,0]] # to RGB image
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

    rel_pairs, show_obj_names = [], None
    boxes = boxlist.bbox.cpu()

    if mode == 'annotation':
        if type(dataset).__name__ in ['VG150Dataset', 'COCOCaptionSceneGraphDataset', 'VGCaptionSceneGraphDataset']:
            # object names
            obj_names = []
            for ind, x in enumerate(boxlist.get_field('labels')):
                obj_names.append(f"{dataset.ind_to_classes[x]}({ind})")

            # relations
            if 'relation' in boxlist.extra_fields.keys():
                relation_map = boxlist.get_field('relation')
                rel_pairs = relation_map.nonzero()
                rel_labels = relation_map[rel_pairs[:,0], rel_pairs[:,1]]
            show_obj_names = obj_names
        elif type(dataset).__name__ in ['UnboundedVGSceneGraphDataset']:
            # object names
            obj_names = []
            for ind, x in enumerate(boxlist.get_field('box_labels')):
                obj_names.append(f"{x}({ind})")

            # relations
            if 'relation' in boxlist.extra_fields.keys():
                relation_map = boxlist.get_field('relation')
                rel_pairs = relation_map.nonzero()
                rel_labels = boxlist.get_field('relation_labels_dict')
            show_obj_names = obj_names
    elif mode == 'prediction':
        VG150_BASE_OBJ_CATEGORIES = ['__background__', 'tile', 'drawer', 'men', 'railing', 'stand', 'towel', 'sneaker', 'vegetable', 'screen', 'vehicle', 'animal', 'kite', 'cabinet', 'sink', 'wire', 'fruit', 'curtain', 'lamp', 'flag', 'pot', 'sock', 'boot', 'guy', 'kid', 'finger', 'basket', 'wave', 'lady', 'orange', 'number', 'toilet', 'post', 'room', 'paper', 'mountain', 'paw', 'banana', 'rock', 'cup', 'hill', 'house', 'airplane', 'plant', 'skier', 'fork', 'box', 'seat', 'engine', 'mouth', 'letter', 'windshield', 'desk', 'board', 'counter', 'branch', 'coat', 'logo', 'book', 'roof', 'tie', 'tower', 'glove', 'sheep', 'neck', 'shelf', 'bottle', 'cap', 'vase', 'racket', 'ski', 'phone', 'handle', 'boat', 'tire', 'flower', 'child', 'bowl', 'pillow', 'player', 'trunk', 'bag', 'wing', 'light', 'laptop', 'pizza', 'cow', 'truck', 'jean', 'eye', 'arm', 'leaf', 'bird', 'surfboard', 'umbrella', 'food', 'people', 'nose', 'beach', 'sidewalk', 'helmet', 'face', 'skateboard', 'motorcycle', 'clock', 'bear']

        if 'VG' in type(dataset).__name__:
            obj_names = []
            # proposal stage
            for ind, x in enumerate(boxlist.get_field('labels')):
                obj_names.append(f"{dataset.categories()[int(x)]}({ind})")
                if dataset.categories()[int(x)] not in VG150_BASE_OBJ_CATEGORIES:
                    obj_names[-1] = f"*{obj_names[-1]}*"

            # # final stage
            # for ind, x in enumerate(boxlist.get_field('pred_labels')):
            #     obj_names.append(f"{dataset.ind_to_classes[x]}({ind})")

            show_obj_names = obj_names
            if 'rel_pair_idxs' in boxlist.extra_fields.keys():
                rel_pairs = boxlist.get_field('rel_pair_idxs')[:rel_num]
                rel_labels = boxlist.get_field('pred_rel_labels')[:rel_num]

                # only show objs with rels
                keep_inds = rel_pairs.view(-1).unique().tolist()
                boxes = boxes[keep_inds]
                show_obj_names = [obj_names[i] for i in keep_inds]

    # show image with boxes
    plt.imshow(img_tensor)
    for ind, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1,y1), x2-x1+1, y2-y1+1, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        if show_obj_names is not None:
            txt = plt.text(x1-10, y1-10, show_obj_names[ind], color='black')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    if 'filename' in boxlist.extra_fields:
        mode = f"{mode}: {boxlist.get_field('filename')}"
    plt.title(mode)

    # print relations
    rel_strs = ''
    for i, rel in enumerate(rel_pairs): # print relation triplets
        if isinstance(rel_labels, dict):
            rel_strs += f"{obj_names[rel[0]]} ---{rel_labels[(rel[0].item(), rel[1].item())]}----> {obj_names[rel[1]]}\n"
        else:
            rel_strs += f"{obj_names[rel[0]]} ---{dataset.ind_to_predicates[rel_labels[i]]}----> {obj_names[rel[1]]}\n"
    print(rel_strs)

    plt.gca().yaxis.set_label_position("right")
    plt.ylabel(rel_strs, rotation=0, labelpad=-20, fontsize=9, loc='top', color='r')
    plt.show()

