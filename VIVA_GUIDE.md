# GAN viva guide

## Introduction
"Ham conditional GAN se old photos restore karte hain. Generator damaged photo aur mask se restored image banata hai. Discriminator real clean image aur generated image ke patches judge karta hai. Dono adversarially train hote hain. Streamlit frontend upload, compare aur download provide karta hai."

Only say "trained" after actual training completes and the results are checked.

## Code flow
1. `PhotoPairs` clean crop leta hai. `damage()` scratches/fading/noise add karke damaged input aur clean target ka pair banata hai.
2. `Pix2PixGenerator` encoder se features nikalta hai aur decoder se RGB restore karta hai. Skip connections detail bachate hain. Residual output input mein correction add karta hai.
3. `PatchDiscriminator` condition aur real/generated image saath dekhta hai aur patches ke logits deta hai.
4. `gan_train_step` pehle D update, phir G update karta hai.
5. `train` epochs repeat karta hai, validation aur checkpoints save karta hai.
6. `restore_automatic` narrow scratch mask detect karta hai. `neural_restore` mein sirf Generator chalta hai; overlapping tiles memory limit karte hain. Selected scratches original local context se inpaint hote hain, phir intact pixels se estimated GAN tone correction milti hai. Yeh OpenCV stage GAN se alag hai.
7. `App.py` mein user sirf photo upload karta hai. App automatic mask, processing size aur trained Generator choose karta hai, phir original/restored images aur ek Download button dikhata hai. Technical settings aur graphs frontend se hata diye gaye hain.

## Main training step
- `fake = generator(inp)`: damaged input ko restore karne ki current attempt.
- `zero_grad()`: pichhle gradients clear.
- `discriminator(inp, target)`: actual clean photo; correct label 1.
- `discriminator(inp, fake.detach())`: generated photo; correct label 0. detach se is D step mein G update nahi hota.
- `d_loss.backward(); optimizer_d.step()`: D ko improve karna.
- `discriminator.requires_grad_(False)`: G step mein D ki weights freeze. Gradient D ke operations se G tak phir bhi flow karta hai.
- `BCE(fake_score, ones)`: G chahta hai D output ko real samjhe.
- Scratch-weighted L1: scratch pixels ko 9x importance; total weights se normalize. Chhote damaged regions global average mein ignore nahi hote.
- Edge L1: horizontal/vertical neighbouring pixel differences match karna, taaki excessive smoothing reduce ho.
- `g_loss = adversarial + 100 * (reconstruction + 0.15 * edge)`: realism, correct content aur edges.
- `g_loss.backward(); optimizer_g.step()`: G ko improve karna.

## Viva questions
| Question | Answer |
|---|---|
| GAN kya hai? | G photo banata hai, D real/fake judge karta hai; dono compete karke learn karte hain. |
| Conditional kyon? | Output ko particular damaged input se related rehna hai. |
| U-Net aur GAN dono kaise? | U-Net Generator ki architecture hai. G + trained D milkar GAN bante hain. |
| Channels? | G: RGB 3 + mask 1 = 4. D: condition 4 + candidate RGB 3 = 7. |
| PatchGAN? | Local patches judge karta hai. 256x256 input par 30x30 logits; nominal convolutional receptive field 70x70. |
| BCEWithLogits? | Stable real/fake classification loss; sigmoid calculation included hai. |
| L1 kyon? | Correct clean image se pixel-wise match. |
| Factor 100? | Common Pix2Pix starting weight for strong reconstruction; tune kar sakte hain. |
| Adam? | Gradients se weights update; G/D ke separate optimizers. |
| GroupNorm? | Small batch sizes ke liye normalization. |
| Inference mein D? | Nahi, sirf G. D training judge hai. |
| Paired data? | Clean DIV2K ko synthetically damage karte hain; original target rehti hai. |
| PSNR/SSIM? | Known clean reference se similarity; historical truth prove nahi hoti. |
| Colorization? | Separate optional pretrained ECCV16 stage; original colors guaranteed nahi. |
| Limitations? | Severe missing areas, false masks, unseen damage, invented detail. |

Show both network classes, gan_train_step, actual G/D loss curves and held-out comparisons. OpenCV or old supervised weights ko GAN output mat batana. Compact Generator original eight-level Pix2Pix ka adaptation hai.

Epoch 10 active hai: 800 training / 100 validation sources, four extra epochs from epoch 6. 30% paired B&W examples mein input aur target dono grayscale hain; 25% training pairs estimated masks use karte hain. Colorization nahi sikhayi. Automatic-mask validation comparison: PSNR 21.12 -> 21.14 dB; SSIM 0.801 -> 0.813; scratch MAE 45.52 -> 36.26. Healthy contrast ho to GAN correction automatically kam hoti hai, taaki intact texture preserve ho. Separate category checks mein faded, scratched, damaged aur B&W sabke average PSNR/SSIM improve hue. Validation detector tuning mein bhi use hua, isliye independent test nahi hai. Large missing areas mein original detail recover hone ka claim nahi karna.

