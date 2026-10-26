# GAN viva guide

## Final app: kaunsa model kya karta hai?

- Microsoft ka pretrained U-Net scratch/damage mask predict karta hai. Yeh supervised detector hai, GAN nahi.
- Pretrained LaMa generator selected damage ko context se fill karta hai. LaMa adversarial/GAN aur perceptual losses se train hua tha; hamne iske weights download kiye hain, locally LaMa train nahi kiya.
- Hamara `model.py` wala Pix2Pix G/D actually epoch 40 tak DIV2K par train hua. Final app low-contrast images mein is G ko optional tone correction ke liye use karti hai; accepted normal-contrast lady portrait ko extra tone/face enhancement nahi diya jata.
- CelebA ka full training job **start nahi hua**, aur koi CelebA-trained checkpoint save/deploy nahi hua. Sirf 64-image speed probe ka temporary model discard kiya gaya. User accepted result ke baad dataset workflow cancel hua. GFPGAN enhancement enabled nahi hai.
- UI simple hi hai: upload, automatic repair, before/after aur download. User photographs external server par upload nahi hote.

Pretrained models ko khud trained models mat bolna. `THIRD_PARTY_MODELS.md` mein sources/licences aur `setup_repair_models.py` mein reproducible model installation hai. Neeche local Pix2Pix training ka explanation hai.

## Introduction
"Final photo repair ke liye learned scratch detector aur pretrained LaMa GAN use hote hain. Hamne ek separate Pix2Pix conditional GAN bhi train kiya: Generator damaged photo aur mask se clean target predict karta hai, Discriminator real/generated patches judge karta hai. Yeh local model low-contrast tone correction aur baseline comparison ke liye available hai. Streamlit upload, compare aur download provide karta hai."

Only say "trained" after actual training completes and the results are checked.

## Code flow
1. `PhotoPairs` clean crop leta hai. `damage()` scratches/fading/noise add karke damaged input aur clean target ka pair banata hai.
2. `Pix2PixGenerator` encoder se features nikalta hai aur decoder se RGB restore karta hai. Skip connections detail bachate hain. Residual output input mein correction add karta hai.
3. `PatchDiscriminator` condition aur real/generated image saath dekhta hai aur patches ke logits deta hai.
4. `gan_train_step` pehle D update, phir G update karta hai.
5. `train` epochs repeat karta hai, validation aur checkpoints save karta hai.
6. `restore_automatic` installed selected backend choose karta hai: learned detector, LaMa fill, aur zaroorat ho to local Pix2Pix tone correction. `neural_restore` local G ke overlapping tiles chalata hai. Explicit `backend='legacy'` se purana Pix2Pix/OpenCV comparison available hai.
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

Epoch 40 active hai: epoch 10 se 30 extra epochs, RTX 4050 par saari 800 original HR training photos har epoch mein use hui. Har photo se random 256×256 crop; poori photo ke saare pixels har epoch nahi. 100 validation sources training se alag hain. `fast` clean resized duplicates hain, defective LR dataset nahi.

Realistic curriculum mein faded/grey/transparent bent scratches, spots, intact aur fade-only examples hain. 30% paired B&W examples mein input aur target dono grayscale hain. Final phase mein 75% condition masks detector se aate hain; actual damage mask alag rakha hai aur sirf loss weighting ke liye use hota hai. Isliye detector miss kare tab bhi G ko scratch repair ka strong learning signal milta hai. Checkpoint validation predicted masks se hoti hai. D aur G dono resume/train hue, Adam LR 0.0001, float32 aur batch 4.

Matched HR automatic-mask validation: G PSNR 22.67 -> 25.81 dB, SSIM 0.8615 -> 0.8944. New grey-crease set mein scratch MAE lagbhag 19% kam; old mixed set ka PSNR same aur SSIM thoda lower hai. Separate faded/scratched/damaged/B&W category averages improve hue. Actual portrait visually inspect aur app download verify kiya; small spots aur border damage abhi bache hain. Validation selection/tuning mein use hua, isliye independent test nahi. Colorization disabled hai; missing facial detail ka original sach recover hone ka claim nahi karna. Full evidence: [TRAINING_REPORT.md](TRAINING_REPORT.md).

