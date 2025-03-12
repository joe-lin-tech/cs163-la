#import "@preview/polylux:0.3.1": *
#import "@preview/fletcher:0.4.5" as fletcher: node, edge
#import "@preview/cetz:0.3.0": *
#import "@preview/suiji:0.3.0": *
// #import "@preview/cetz-plot:0.1.0": plot

#import themes.simple: *
#import themes.university: *

#set text(font: "PT Sans")
#set enum(numbering: "1.a.i.")

#show: university-theme.with(
  color-a: rgb(39, 116, 174),
  color-b: rgb(255, 209, 0),
  color-c: rgb(255, 255, 255)
)


#matrix-slide[
  = Week 10 Discussion 1A
  #v(2em)

  Joe Lin and Krystof Latka \
  _Learning Assistants_

  December 6, 2024
]

#focus-slide[
  === Final Review
]

#slide[
  = Recurrent Neural Networks
  #grid(columns: (auto, auto), )[
    A *Recurrent Neural Network* processes a sequence of input vectors by applying a recurrence formula at every time step.

    Contains a:
    - Hidden State $h$

    $ h_t = f_W (h_(t - 1), x_t) $
  ][
    #align(center)[
      #fletcher.diagram(
        node-inset: 15pt,
        {
          node($y$, enclose: ((-0.1, 0.25), (0.1, 0)), shape: "rect", stroke: blue + 0.5pt, fill: blue.lighten(80%))
          node("RNN", enclose: ((-0.5, 1.25), (0.5, 1.0)), shape: "rect", stroke: red + 0.5pt, fill: red.lighten(80%))
          node($x$, enclose: ((-0.1, 2.25), (0.1, 2.0)), shape: "rect", stroke: green + 0.5pt, fill: green.lighten(80%))
          node((-1.0, 1.0), "")
          edge((0, 1.0), (0, 0.25), "-|>")
          edge((0, 2.25), (0, 1.25), "-|>")
          edge((0.7, 0.9), (0.7, 1.35), "-|>", bend: 30deg)
        }
      )
    ]
  ]
]

#slide[
  = Recurrent Neural Networks
  #align(center)[
    #fletcher.diagram(
      node-inset: 15pt,
      {
        let x = ($x_1$, $x_2$, $x_3$)
        let h = ($h_1$, $h_2$, $h_3$)
        let y = ($y_1$, $y_2$, $y_3$)
        for i in range(1, 4) {
          node(y.at(i - 1), enclose: ((-0.1 + (i - 1), 0.25), (0.1 + (i - 1), 0)), shape: "rect", stroke: blue + 0.5pt, fill: blue.lighten(80%))
          node(h.at(i - 1), enclose: ((-0.1 + (i - 1), 1.25), (0.1 + (i - 1), 1.0)), shape: "rect", stroke: red + 0.5pt, fill: red.lighten(80%))
          node(x.at(i - 1), enclose: ((-0.1 + (i - 1), 2.25), (0.1 + (i - 1), 2.0)), shape: "rect", stroke: green + 0.5pt, fill: green.lighten(80%))
          if (i == 3) {
            edge(((i - 1), 1.0), ((i - 1), 0.25), "-|>", $W_(y h)$)
            edge(((i - 1), 2.25), ((i - 1), 1.25), "-|>", $W_(h x)$)
          } else {
            edge(((i - 1), 1.0), ((i - 1), 0.25), "-|>")
            edge(((i - 1), 2.25), ((i - 1), 1.25), "-|>")
          }
          if (i == 2) {
            edge((i - 2, 1.125), (i - 1, 1.125), "-|>")
          }
          if (i == 3) {
            edge((i - 2, 1.125), (i - 1, 1.125), "-|>", $W_(h h)$)
          }
        }
      }
    )
  ]
]

#slide[
  = Attention
  In basic *attention*, we have the following inputs:
  - Input vectors $X in RR^(N_X times D_X)$
  - Key matrix $W_K in RR^(D_X times D_Q)$
  - Query vectors $Q in RR^(N_Q times D_Q)$
  - Value matrix $W_V in RR^(D_X times D_V)$

  The goal is to use the query vectors and learn to attend to the input vectors and return a relevant combination of those vectors as output.

  How do we do this? #uncover("2-")[#text(blue)[Utilize Key and Value vectors.]]
]

#slide[
  = Attention
  $ K = X W_K in RR^(N_X times D_Q) $
  What did we do here? #uncover("2-")[#text(blue)[Learned a linear transformation $W_K$ to obtain a set of keys from the input vectors.]]

  #pause #pause

  $ V = X W_V in RR^(N_X times D_V) $
  What did we do here? #uncover("4-")[#text(blue)[Learned a linear transformation $W_V$ to obtain a set of values from the input vectors.]]

  #pause #pause

  Essentially, we learn the keys and values associated with $X$.
]

#slide[
  = Attention
  $ E = (Q K^T) / sqrt(D_Q) in RR^(N_Q times N_X) " " " " " " " " " " " " " " " " E_(i, j) = (Q_i dot K_j) / sqrt(D_Q) $
  What did we do here? #uncover("2-")[#text(blue)[Performed a scaled dot product, where $E_(i, j)$ tells us how similar query $Q_i$ and key $K_j$ are.]]

  #pause #pause

  #align(center)[```py A = softmax(E, dim=1)```]
  What did we do here? #uncover("4-")[#text(blue)[Normalize scores such that $A_(i, j)$ now tells us the similarity of query $Q_i$ and key $K_j$ in _proportion_ to all key vectors.]]
]

#slide[
  = Attention
  $ Y = A V in RR^(N_Q times N_V) $
  What did we do here? #uncover("2-")[#text(blue)[Based on attention scores, output a linear combination of value vectors.]]

  $ Y = "softmax"((Q K^T) / sqrt(D_Q)) V $
]

#slide[
  = Self-Attention
  In *self-attention*, we have the following inputs:
  - Input vectors $X in RR^(N_X times D_X)$
  - Key matrix $W_K in RR^(D_X times D_Q)$
  - Query matrix $W_Q in RR^(D_X times D_Q)$
  - Value matrix $W_V in RR^(D_X times D_V)$

  Primary change is to learn to obtain a set of query vectors (one for every input).
]

// #slide[
//   = Self-Attention
//   How does permuting the input vectors affect the outputs of the self-attention layer?
//   #uncover("2-")[
//     #enum(numbering: "a.",
//       [It has no effect.],
//       [Permuting input vectors completely changes the attention operations.],
//       [#alternatives-cases(("1, 2", "3"), case => [
//         #set text(fill: blue) if case == 1
//         Outputs are correspondingly permuted.
//       ])]
//     )
//   ]
// ]

#slide[
  = Transformer
  Consider a transformer block (with one self-attention layer and one MLP layer) that processes $N$ tokens of dimension $D$. How many parameters (excluding biases and layer norm) are in this block?

  #pause

  #text(blue)[Self-attention layer has $3D^2$ parameters and MLP layer has $D^2$ parameters, so there are $4D^2$ total parameters.]
]

#slide[
  = Vision Transformer
  1. Treat image patches as tokens.
  2. Apply linear projection to obtain $D$-dimensional patch embeddings.
  3. Add positional encoding to embeddings.
  4. Input into regular transformer architecture.
]

#slide[
  = Hierarchical ViT
  #image("swin-transformer.png", height: 80%)
]

#slide[
  = Hierarchical ViT
  What's the purpose of pipeline displayed in part (d) and how does it accomplish this?
  #uncover("2-")[#text(blue)[Enable transformer to learn visual features at multiple scales. Uses patch merging to reduce resolution.]]

  #pause
  #pause

  How does Swin Transformer avoid costly global attention computations?
  #uncover("4-")[
    #text(blue)[
      Uses windowed attention instead. Size of attention matrices can be reduced:
      $ H^2 W^2 -> M^2 H W $
    ]
  ]
]

#slide[
  = Region-Based CNN
  #image("rcnn.jpg")

  Use *selective search* algorithm to propose region of interests (RoIs). Resize image for CNN compatability and classify.
]

#slide[
  = Region-Based CNN
  #grid(columns: 2, gutter: 32pt, align: horizon)[
    #image("rcnn-bbox.jpg")
  ][
    #text(red)[Problem: How to handle incorrectly sized *region proposals*?]

    #pause

    #text(green)[Solution: Learn to predict *bounding box transforms*.]

    $ b_x = p_x + p_w t_x " " " " " " b_y = p_y + p_h t_y \
    b_w = p_w"exp"(t_w) " " " " " " b_h = p_h"exp"(t_h) $
  ]
]

#slide[
  = Fast Region-Based CNN
  #grid(columns: 2, gutter: 32pt)[
    #image("fast-rcnn.jpg", width: 100%)
  ][
    #text(red)[Problem: Very slow to feed every proposed region into entire CNN.]
    
    #pause

    #text(green)[Solution: Pass entire image into CNN, project RoI, and apply *differentiable cropping*.]
  ]

  After projecting regions of interest (RoIs) to feature map, how do we select features to predict class and bounding box transforms on?
]

#slide[
  = Region of Interest Align
  #grid(columns: 2, gutter: 32pt)[
    #fletcher.diagram(
      node-inset: 0pt,
      {
        for i in range(0, 9) {
          edge((i / 2, 0), (i / 2, 4))
          edge((0, i / 2), (4, i / 2))
        }
        let c = (2.4, 2.1)
        let lu = (c.at(0) - 0.75, c.at(1) - 1)
        let rl = (c.at(0) + 0.75, c.at(1) + 1)
        edge((lu.at(0), lu.at(1)), (rl.at(0), lu.at(1)), stroke: blue.lighten(50%) + 4pt)
        edge((lu.at(0), rl.at(1)), (rl.at(0), rl.at(1)), stroke: blue.lighten(50%) + 4pt)
        edge((lu.at(0), lu.at(1)), (lu.at(0), rl.at(1)), stroke: blue.lighten(50%) + 4pt)
        edge((rl.at(0), lu.at(1)), (rl.at(0), rl.at(1)), stroke: blue.lighten(50%) + 4pt)
        edge((lu.at(0), (lu.at(1) + rl.at(1)) / 2), (rl.at(0), (lu.at(1) + rl.at(1)) / 2), stroke: blue.lighten(50%) + 1pt, "dashed")
        edge(((lu.at(0) + rl.at(0)) / 2, lu.at(1)), ((lu.at(0) + rl.at(0)) / 2, rl.at(1)), stroke: blue.lighten(50%) + 1pt, "dashed")
        for c in range(1, 6) {
          for r in range(1, 6) {
            if (c == 3 or r == 3) {
              continue
            }
            node((lu.at(0) + 0.25 * c, lu.at(1) + 1 / 3 * r), circle(radius: 4pt, fill: blue.lighten(60%)))
          }
        }
      }
    )
  ][
    How do we obtain features at sampling points in a differentiable manner? #uncover("2-")[#text(blue)[Bilinear interpolation.]]

    _Note the blue rectangle is a projected region of interest on feature map._
  ]
]

#slide[
  = Region of Interest Align
  #fletcher.diagram(
    node-inset: 0pt,
    {
      for i in range(0, 9) {
        edge((i / 2, 0), (i / 2, 4))
        edge((0, i / 2), (4, i / 2))
      }
      let c = (2.4, 2.1)
      let lu = (c.at(0) - 0.75, c.at(1) - 1)
      let rl = (c.at(0) + 0.75, c.at(1) + 1)
      edge((lu.at(0), lu.at(1)), (rl.at(0), lu.at(1)), stroke: blue.lighten(50%) + 4pt)
      edge((lu.at(0), rl.at(1)), (rl.at(0), rl.at(1)), stroke: blue.lighten(50%) + 4pt)
      edge((lu.at(0), lu.at(1)), (lu.at(0), rl.at(1)), stroke: blue.lighten(50%) + 4pt)
      edge((rl.at(0), lu.at(1)), (rl.at(0), rl.at(1)), stroke: blue.lighten(50%) + 4pt)
      edge((lu.at(0), (lu.at(1) + rl.at(1)) / 2), (rl.at(0), (lu.at(1) + rl.at(1)) / 2), stroke: blue.lighten(50%) + 1pt, "dashed")
      edge(((lu.at(0) + rl.at(0)) / 2, lu.at(1)), ((lu.at(0) + rl.at(0)) / 2, rl.at(1)), stroke: blue.lighten(50%) + 1pt, "dashed")
      for c in range(1, 6) {
        for r in range(1, 6) {
          if (c == 3 or r == 3) {
            continue
          }
          node((lu.at(0) + 0.25 * c, lu.at(1) + 1 / 3 * r), circle(radius: 4pt, fill: blue.lighten(60%)))
        }
      }
      let offset = (5, 0.45)
      for i in range(0, 3) {
        edge((offset.at(0) + i, offset.at(1)), (offset.at(0) + i, offset.at(1) + 2))
        edge((offset.at(0), offset.at(1) + i), (offset.at(0) + 2, offset.at(1) + i))
      }
      edge((1.5, 1), (offset.at(0), offset.at(1)), "dashed")
      edge((2.5, 1), (offset.at(0) + 2, offset.at(1)), "dashed")
      edge((1.5, 2), (offset.at(0), offset.at(1) + 2), "dashed")
      edge((2.5, 2), (offset.at(0) + 2, offset.at(1) + 2), "dashed")
      node((offset.at(0) + 0.5, offset.at(1) + 0.5), circle(radius: 6pt, fill: gray.lighten(60%)))
      node((offset.at(0) + 0.5, offset.at(1) + 0.3), text(size: 14pt, gray.lighten(60%))[$10$])
      node((offset.at(0) + 1.5, offset.at(1) + 0.5), circle(radius: 6pt, fill: gray.lighten(60%)))
      node((offset.at(0) + 1.5, offset.at(1) + 0.3), text(size: 14pt, gray.lighten(60%))[$0$])
      node((offset.at(0) + 0.5, offset.at(1) + 1.5), circle(radius: 6pt, fill: gray.lighten(60%)))
      node((offset.at(0) + 0.5, offset.at(1) + 1.3), text(size: 14pt, gray.lighten(60%))[$10$])
      node((offset.at(0) + 1.5, offset.at(1) + 1.5), circle(radius: 6pt, fill: gray.lighten(60%)))
      node((offset.at(0) + 1.5, offset.at(1) + 1.3), text(size: 14pt, gray.lighten(60%))[$100$])
      node((offset.at(0) + calc.rem(lu.at(0) * 2 + 0.5, 1), offset.at(1) + calc.rem(lu.at(1) * 2 + 2 / 3, 1)), circle(radius: 6pt, fill: blue.lighten(60%)))
    }
  )
]

#slide[
  = Region of Interest Align
  Suppose gray dots are at coordinates $(3, 2), (4, 2), (3, 3), (4, 3)$ (top left, top right, bottom left, bottom right) and blue dot is at $(3.3, 2.4)$. Compute the interpolated feature.

  #pause
  
  #text(blue)[
    $ 10 dot 0.7 dot 0.6 + 0 dot 0.3 dot 0.6 + 10 dot 0.4 dot 0.7 + 100 dot 0.3 + 0.4 \
    = 4.2 + 2.8 + 12 \
    = 19 $
  ]
]

#slide[
  = Faster Region-Based CNN
  #grid(columns: 2, gutter: 32pt)[
    #image("faster-rcnn.png", height: 80%)
  ][
    #text(red)[Problem: Runtime dominated by region proposals (e.g. selective search).]
    
    #pause

    #text(green)[Solution: Add a *region proposal network* to predict region proposals.]
  ]
]

// #slide[
//   = Region Proposal Networks
//   Suppose a feature map $f in RR^(C times H times W)$ is fed into a region proposal network with $K = 5$ anchor points per position. Let $C = 128, H = 10, W = 10$.

//   - How many anchors are there in total? #uncover("2-")[#text(blue)[$5 dot 10 dot 10 = 500$ anchors.]]
//   - Suppose we have two separate RPN classification and regression branches, what are the input and output channels for these two CNNs? #uncover("3-")[#text(blue)[Classification -- $128 -> 5$, Regression -- $128 -> 5 dot 4 = 20$]]
// ]

#slide[
  = Region Proposal Networks
  #grid(columns: 2)[
    Start with $K$ anchors per position (remember that the input is a feature map $f$).

    What's the purpose of a region proposal network? #uncover("2-")[#text(blue)[The goal of a region proposal network is to:
    1. Classify each anchor as region of interest (RoI) or background.
    2. Regress anchor box offset transforms.]]
  ][
    #align(horizon)[#image("anchors.png", height: 50%)]
  ]
]

#slide[
  = Non-Maximum Suppression
  #text(red)[Problem: What do we do with overlapping detections?]

  #pause

  #text(green)[Solution: Use *non-maximum suppression (NMS)* to filter predictions.]

  1. Sort predictions by confidence.
  2. For each remaining prediction, loop through and discard predictions with IoU above some threshold.
    - What does this step do? #uncover("2-")[#text(blue)[Removes predictions that have high overlap.]]
]

#slide[
  = Detection Metrics
  Consider the following ground truth and prediction maps. Suppose there are $3$ classes. Compute the mean IoU and DICE scores.
  #grid(columns: (50%, 50%), align: center + horizon)[
    #let inputs = (($2$, $1$, $1$, $2$), ($1$, $1$, $1$, $1$), ($0$, $0$, $0$, $0$), ($2$, $2$, $0$, $0$),)
    #canvas(length: 3cm, {
      import draw: *

      set-style(
        mark: (fill: black, scale: 2),
        stroke: (thickness: 0.4pt, cap: "round"),
        content: (padding: 1pt)
      )

      grid((0, 0), (2, 2), step: 0.5)
      for c in range(inputs.at(0).len()) {
        for r in range(inputs.len()) {
          content((c * 0.5 + 0.25, r * 0.5 + 0.25), align(center + horizon)[#inputs.at(r).at(c)])
        }
      }
    })
    *Ground Truth*
  ][
    #let inputs = (($2$, $2$, $2$, $2$), ($1$, $1$, $1$, $2$), ($0$, $2$, $1$, $0$), ($1$, $2$, $0$, $1$),)
    #canvas(length: 3cm, {
      import draw: *

      set-style(
        mark: (fill: black, scale: 2),
        stroke: (thickness: 0.4pt, cap: "round"),
        content: (padding: 1pt)
      )

      grid((0, 0), (2, 2), step: 0.5)
      for c in range(inputs.at(0).len()) {
        for r in range(inputs.len()) {
          content((c * 0.5 + 0.25, r * 0.5 + 0.25), align(center + horizon)[#inputs.at(r).at(c)])
        }
      }
    })
    *Predictions*
  ]
]

#slide[
  = Detection Metrics
  #text(blue)[To compute scores, we first calculate for each class, then take the mean.]
  #grid(columns: 2, gutter: 48pt)[
    #text(blue)[
      IoU:
      - Class 0 -- $3 / 6 = 1 / 2$
      - Class 1 -- $3 / 9 = 1 / 3$
      - Class 2 -- $3 / 8$
      - Mean -- $29 / 72$
    ]
  ][
    #text(blue)[
      DICE:
      - Class 0 -- $6 / 9 = 2 / 3$
      - Class 1 -- $6 / 12 = 1 / 2$
      - Class 2 -- $6 / 11$
      - Mean -- $113 / 198$
    ]
  ]
]

#slide[
  = Semantic Segmentation
  Typically consists of a encoder-decoder architecture.
  - Encoder uses convolutions to *downsample*.
  - Decoder *upsamples* with *transposed convolutions* to obtain a semantic map at the original resolution.
]

#slide[
  = Mask Region-Based CNN
  Add a mask prediction head to typical R-CNN framework.
  #align(center)[#image("mask-rcnn.png")]
]

#slide[
  = Generative Models
  What are we trying to learn with a:
  - *Discriminative Model* -- #uncover("2-")[#text(blue)[$p(y | x)$]]
    - For MiniPlaces, you trained a predictor to learn a probability distribution over the classes given an image.
  - *Generative Model* -- #uncover("3-")[#text(blue)[$p(x)$]]
    - Generate an image.
  - *Conditional Generative Model* -- #uncover("4-")[#text(blue)[$p(x | y)$]]
    - Generate an image given a certain text.
]

#slide[
  = Generative Models
  Match the following generative models with their properties.
  #grid(columns: (50%, 45%), row-gutter: 32pt, column-gutter: 18pt,
    [a. Autoregressive Models], align(right)[explicit density, approximate density, variational],
    [b. Variational Autoencoders], align(right)[implicit density, direct],
    [c. Diffusion Models], align(right)[explicit density, approximate density, markov chain],
    [d. Generative Adversarial Networks], align(right)[explicit density, tractable density]
  )
]

#slide[
  = Generative Models
  Match the following generative models with their properties.
  #grid(columns: (50%, 45%), row-gutter: 32pt, column-gutter: 18pt,
    [#text(purple)[a. Autoregressive Models]], align(right)[#text(red)[explicit density, approximate density, variational]],
    [#text(red)[b. Variational Autoencoders]], align(right)[#text(green)[implicit density, direct]],
    [#text(blue)[c. Diffusion Models]], align(right)[#text(blue)[explicit density, approximate density, markov chain]],
    [#text(green)[d. Generative Adversarial Networks]], align(right)[#text(purple)[explicit density, tractable density]]
  )
]

#slide[
  = Variational Autoencoders
  Trained by maximizing the *variational lower bound* (often referred to as *ELBO*).
  $ EE_(z tilde q_phi (z | x)) [log p_theta (x | z)] - D_(K L) (q_phi (z | x), p(z)) $

  What's the first term? #uncover("2-")[#text(blue)[$EE_(z tilde q_phi (z | x)) [log p_theta (x | z)]$ ensures that original input data is probable in decoder output distribution.]]

  What's the second term? #uncover("4-")[#text(blue)[$D_(K L) (q_phi (z | x), p(z))$ ensures that the encoder output distribution is close to the prior distribution $p(z)$.]]
]

#slide[
  = Variational Autoencoders
  Give one similarity and one difference between the standard auto-encoder and the variational auto-encoder.

  #pause

  #grid(columns: 2, gutter: 24pt)[
    #text(blue)[
      Similarities:
      - Encoder-decoder paradigm
      - Latent bottleneck
      - Model $x$ as a function of latents $z$
    ]
  ][
    #text(blue)[
      Differences:
      - Sampling
      - Minimize variational lower bound instead of just reconstruction loss
      - Variational autoencoders are generative
    ]
  ]
]

#slide[
  = Generative Adversarial Networks
  Consider the loss objective used when training GANs.
  $ min_G max_D (EE_(x tilde p_"data") [log(D(x))] + EE_(z tilde p(z))[log(1 - D(G(z)))]) $
  Traditionally, both $D$ and $G$ are randomly initialized. Assume instead that $G$ is randomly initialized, but $D$ is a perfect discriminator.

  #alternatives[How will this affect training?][How will this affect training?][Propose one modification to the loss function to improve the training \ of the GAN under this assumption.]
  #only("2")[#text(blue)[Vanishing gradient for $G$ because the second term is close to 0.]]
  #only("4")[#text(blue)[Any modification that increases gradient of the second term near $0$, for instance using $-log D(G)$ instead, while preserving gradient direction.]]
]

#slide[
  = Diffusion Models
  1. Start with a ground-truth input image.
  2. Iteratively add random noise for $t$ steps.
  3. Given noised image and $t$, learn to iteratively denoise the image.

  #align(center)[#image("diffusion.png", width: 80%)]
]

#slide[
  = Diffusion Models
  *Forward diffusion* process -- adds noise to the data in a controlled manner over a series of time steps, effectively transforming the data distribution into a normal distribution
  - Note that this is _defined_ by us beforehand as follows:
  $ x_t = sqrt(1 - beta_t) dot x_(t - 1) + sqrt(beta_t) dot epsilon $
  *Reverse diffusion* process -- iteratively denoise a sample starting from pure Gaussian noise, resulting in a reconstruction of original data
  - Note that this is learned (typically with a U-Net or Transformer architecture)
]

#slide[
  = Latent Diffusion Models
  Describe how *latent diffusion models* differ from standard diffusion models in terms of input space and computational efficiency.

  #grid(columns: 2, gutter: 24pt)[
    #text(blue)[
      Input Space
      - Standard -- operate directly on high-dimensional data
      - Latent -- operate in a lower-dimensional latent space obtained via an encoder from a pre-trained autoencoder
    ]
  ][
    #text(blue)[
      Efficiency
      - Standard -- computationally expensive, especially for large image resolutions
      - Latent -- lower computational costs while preserving semantic features of the data
    ]
  ]
]
