from transformers import SegformerForSemanticSegmentation

def segformer_model(classes):
    '''
        Build the SegFormer model using the MiT-B3 encoder which has been pretrained on the ImageNet-1K dataset
    '''
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b3',
        num_labels = len(classes)
    )

    return model