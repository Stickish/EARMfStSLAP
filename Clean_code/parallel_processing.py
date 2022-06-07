
# Run vectorize data in parallell
import numpy as np
import itertools

def fast_vectorize_data(vectorizer, orders, n_neg=1):
    embeddings = []

    # Loop through the orders
    for order in orders:
        
        combs = itertools.combinations(order, 2) # Create the combinations on strings, we think that this is faster
        for set_2 in combs:

            a = vectorizer.fast_transform(set_2[0]) # Anchor
            p = vectorizer.fast_transform(set_2[1]) # Positive

            # Negative sampling 
            for _ in range(n_neg):
                
                n = str(np.random.choice(vectorizer.unique_articles)) # Negative
                # We want the nagative sample to no be in the current order
                n_in_order = n in order
                while n_in_order:
                    n = str(np.random.choice(vectorizer.unique_articles))
                    n_in_order = n in order

                n = vectorizer.fast_transform(n)
                # We count both the anchor and positive as anchors
                embeddings.append([a, p, n])
                embeddings.append([p, a, n])

    return embeddings

def split_into_rows(n_workers, orders):
    order_list = []
    orders2 = list(orders['Articles'])
    
    if len(orders2) % n_workers == 0:
        # Dividable
        size = len(orders2) // n_workers
        
        for i in range(n_workers):
            row = orders2[i*size:(i+1)*size]
            order_list.append(row)

        return order_list

    else:
        n_over = len(orders2) % n_workers
        size = len(orders2) // n_workers
        
        for i in range(n_workers):
            row = orders2[i*size:(i+1)*size]
            order_list.append(row)
        
        for j in range(n_over):
            idx = j+1
            order = orders2[-idx]
            order_list[j].append(order)

        return order_list
        