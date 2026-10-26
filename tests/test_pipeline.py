"""Functional checks; do not substitute for a full training quality evaluation."""
import io
import zipfile
from argparse import Namespace
import numpy as np
import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest
from model import (ROOT, Pix2PixGenerator, PatchDiscriminator, gan_train_step, damage,
                   neural_restore, classical_restore, metric_pair, load_restorer, train)
from prepare_data import safe_extract


def test_damage_and_baseline():
    image = np.full((64,80,3),140,np.uint8)
    a,mask = damage(image,42)
    b,mask2 = damage(image,42)
    assert np.array_equal(a,b) and np.array_equal(mask,mask2)
    assert mask.any() and a.shape == image.shape
    assert classical_restore(a,mask).shape == image.shape
    assert metric_pair(image,image)[1] == pytest.approx(1)


def test_backward_checkpoint_and_tiling(tmp_path):
    torch.set_num_threads(2)
    net = Pix2PixGenerator(base=8)
    inp = torch.rand(1,4,32,40)
    pred = net(inp)
    loss = torch.nn.functional.l1_loss(pred,torch.rand_like(pred))
    loss.backward()
    assert torch.isfinite(loss) and net.head.weight.grad is not None
    path = tmp_path/"checkpoint.pt"
    torch.save({"state_dict":net.state_dict(),"base":8,"model_type":"pix2pix_gan"},path)
    loaded,_ = load_restorer(path)
    image = np.full((45,71,3),120,np.uint8)
    out = neural_restore(loaded,image,np.zeros((45,71),np.uint8),tile=32,overlap=8)
    assert out.shape == image.shape and out.dtype == np.uint8


def test_archive_cannot_escape(tmp_path):
    archive = tmp_path/"unsafe.zip"
    with zipfile.ZipFile(archive,"w") as handle:
        handle.writestr("../outside.txt","bad")
    with pytest.raises(ValueError):
        safe_extract(archive,tmp_path/"data")


def test_frontend_starts():
    app = AppTest.from_file(str(ROOT/"App.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Bring the past into focus."


def test_training_writes_usable_checkpoint(tmp_path):
    for split, name in [("train","train-photo.png"),("val","val-photo.png")]:
        directory = tmp_path/split
        directory.mkdir()
        image = np.random.default_rng(42).integers(0,256,(40,40,3),dtype=np.uint8)
        Image.fromarray(image).save(directory/name)
    args = Namespace(train_dir=str(tmp_path/"train"),val_dir=str(tmp_path/"val"),
        output=str(tmp_path/"run"),epochs=1,samples=2,size=32,batch_size=1,workers=0,
        lr=2e-4,seed=42,patience=2,resume=None,lambda_l1=100.)
    train(args)
    net,checkpoint = load_restorer(tmp_path/"run/best.pt")
    assert checkpoint["epoch"] == 1 and np.isfinite(checkpoint["metrics"]["psnr"])
    assert (tmp_path/"run/history.csv").is_file()
    last = torch.load(tmp_path/"run/last.pt",weights_only=True)
    assert "discriminator" in last and "optimizer_d" in last
    assert checkpoint["model_type"] == "pix2pix_gan"
    args.resume = str(tmp_path/"run/last.pt")
    args.epochs = 2
    train(args)
    assert torch.load(tmp_path/"run/last.pt",weights_only=True)["epoch"] == 2


def test_adversarial_loss_alone_updates_both_networks():
    torch.manual_seed(42)
    generator, discriminator = Pix2PixGenerator(8), PatchDiscriminator(8)
    opt_g = torch.optim.Adam(generator.parameters(),lr=2e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(),lr=2e-4)
    before_g = generator.head.weight.detach().clone()
    before_d = discriminator.layers[0].weight.detach().clone()
    condition = torch.rand(1,4,32,32)*.8+.1
    # lambda_l1=0 proves G gets its learning signal from the GAN, not reconstruction.
    losses = gan_train_step(generator,discriminator,opt_g,opt_d,condition,
                            torch.rand(1,3,32,32),lambda_l1=0)
    assert all(np.isfinite(v) for v in losses.values())
    assert not torch.equal(before_g,generator.head.weight)
    assert not torch.equal(before_d,discriminator.layers[0].weight)
    assert all(p.requires_grad for p in discriminator.parameters())


def test_patchgan_outputs_grid_and_requires_condition():
    discriminator = PatchDiscriminator(8)
    out = discriminator(torch.zeros(1,4,256,256),torch.zeros(1,3,256,256))
    assert out.shape == (1,1,30,30)


def test_supervised_checkpoint_is_not_accepted_as_gan(tmp_path):
    path = tmp_path/"old_unet.pt"
    torch.save({"state_dict":Pix2PixGenerator(8).state_dict(),"base":8},path)
    with pytest.raises(ValueError,match="Pix2Pix GAN"):
        load_restorer(path)
