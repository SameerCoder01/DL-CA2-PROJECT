"""Functional checks; do not substitute for a full training quality evaluation."""
import io
import zipfile
from argparse import Namespace
import numpy as np
import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest
from model import (ROOT, RestorationUNet, damage, restoration_loss, neural_restore,
                   classical_restore, metric_pair, load_restorer, train)
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
    net = RestorationUNet(base=8)
    inp = torch.rand(1,4,32,40)
    pred = net(inp)
    loss = restoration_loss(pred,torch.rand_like(pred),inp[:,3:4])
    loss.backward()
    assert torch.isfinite(loss) and net.head.weight.grad is not None
    path = tmp_path/"checkpoint.pt"
    torch.save({"state_dict":net.state_dict(),"base":8},path)
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
        lr=2e-4,seed=42,patience=2,resume=None)
    train(args)
    net,checkpoint = load_restorer(tmp_path/"run/best.pt")
    assert checkpoint["epoch"] == 1 and np.isfinite(checkpoint["metrics"]["psnr"])
    assert (tmp_path/"run/history.csv").is_file()
