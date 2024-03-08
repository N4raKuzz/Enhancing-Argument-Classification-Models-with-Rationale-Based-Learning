from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')

layer1 = model.encoder.layer[1]

layer2 = model.encoder.layer[2]
query1 = layer1.attention.self.query
query2 = layer2.attention.self.query
#print(layer1)
print(query1.weight.data)
print(query2.weight.data)

