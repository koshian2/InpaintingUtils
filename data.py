import json
import glob
import os
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

def generate_patches(original_patchsize, output_size=None, mask_as_black=True, mode="mosaic"):
    """画像からBounding Boxのある領域を切り出して生成

    Arguments:
        original_patchsize {int} -- 元画像上でのパッチサイズ

    Keyword Arguments:
        output_size {int} -- 出力画像のサイズ。Noneの場合、original_patchsizeと同一。
            original_patchsizeと異なる場合は、リサイズが入る (default: None)
        mask_as_black {bool} -- マスク領域を黒にするかどうか
            Trueなら非マスクが白、マスクが黒。Falseの場合は逆
        mode {string} -- 修正のモード（default:mosaic）
            mosaicならモザイクをかける、whitenなら白抜きする
    """

    jsons = sorted(glob.glob("bbox/*"))
    for f in jsons:
        with open(f) as fp:
            data = json.load(fp)
        for r in data["regions"]:
            bbox = [*r["points"][0].values(), *r["points"][-2].values()]
            # Bounding Boxがパッチからはみ出していないかチェック
            # print(bbox)
            if (bbox[3] - bbox[1]) > original_patchsize:
                margin = (bbox[3] - bbox[1] - original_patchsize) / 2.0
                bbox[1] += margin
                bbox[3] -= margin
            if (bbox[2] - bbox[0]) > original_patchsize:
                margin = (bbox[2] - bbox[0] - original_patchsize) / 2.0
                bbox[0] += margin
                bbox[2] -= margin
            # print(bbox)
            # 縦幅の余り
            pad_Width = (original_patchsize - bbox[2] + bbox[0]) / 2.0
            pad_height = (original_patchsize - bbox[3] + bbox[1]) / 2.0
            # とりあえずCrop範囲を拡大する
            crop_position = [bbox[0] - pad_Width, bbox[1] - pad_height,
                             bbox[2] + pad_Width, bbox[3] + pad_height]
                             
            # 左右のはみ出しチェック
            # 切り出す座標がマイナスになったり、元画像をオーバーしても勝手に黒埋めしてくれるので、
            # 左は溢れているが、右は溢れていないのようなケースのみチェックすれば良い
            original_width, original_height = data["asset"]["size"]["width"], data["asset"]["size"]["height"]            
            if crop_position[0] < 0 and crop_position[2] < original_width:
                shift = min(-crop_position[0], original_width - crop_position[2])
                crop_position[0] += shift
                crop_position[2] += shift
            # 右は溢れているが、左は溢れていないケース
            if crop_position[0] > 0 and crop_position[2] > original_width:
                shift = min(crop_position[0], crop_position[2] - original_width)
                crop_position[0] -= shift
                crop_position[2] -= shift
            # 上は溢れているが、下は溢れていないケース
            if crop_position[1] < 0 and crop_position[3] < original_height:
                shift = min(-crop_position[1], original_height - crop_position[3])
                crop_position[1] += shift
                crop_position[3] += shift
            # 下は溢れているが、上は溢れていないケース
            if crop_position[1] > 0 and crop_position[3] > original_height:
                shift = min(crop_position[1], crop_position[3] - original_height)
                crop_position[1] -= shift
                crop_position[3] -= shift
            # print(crop_position)

            back_color = 255 if mask_as_black else 0
            mask_color = 255 - back_color

            # ファイルが存在しない場合
            if not os.path.exists("images/" + data["asset"]["name"]):
                continue

            with Image.open("images/" + data["asset"]["name"]) as img:
                crop_original = img.crop(crop_position)
                with Image.new("L", (original_patchsize, original_patchsize), color=back_color) as mask:
                    draw = ImageDraw.Draw(mask)
                    rect_pos = [bbox[0] - crop_position[0], bbox[1] - crop_position[1],
                                bbox[2] - crop_position[0], bbox[3] - crop_position[1]]
                    draw.rectangle(rect_pos, fill=mask_color)

                    if output_size is None:
                        output_size = original_patchsize
                    if (output_size != original_patchsize or 
                       output_size != crop_original.width or output_size != crop_original.height or 
                       output_size != mask.width or output_size != mask.height):
                       # たまにずれることがある
                        crop_original = crop_original.resize((output_size, output_size), Image.LANCZOS)
                        mask = mask.resize((output_size, output_size), Image.LANCZOS)

                    if not mask_as_black:
                        flag = ImageOps.invert(mask)
                    else:
                        flag = mask

                    if mode == "mosaic":
                        # ここでモザイクをつける
                        mosaic = crop_original.resize(
                            (crop_original.width // 6, crop_original.height // 6), Image.NEAREST)
                        mosaic = mosaic.filter(ImageFilter.GaussianBlur(1.8))
                        mosaic = mosaic.resize(crop_original.size, Image.NEAREST)

                        # ブレンド
                        degrade = Image.composite(crop_original, mosaic, flag)
                    elif mode == "whiten":
                        # 硬い四角のマスクを作る
                        square_mask = ImageOps.invert(flag)
                        degrade = Image.composite(crop_original, square_mask.convert("RGB"), flag)

                    original_array = np.asarray(crop_original)
                    mask_array = np.asarray(np.expand_dims(mask, -1))
                    degrade_array = np.asarray(degrade)

            yield original_array, degrade_array, mask_array, data

def pack_data_to_pickle(original_patchsize, output_size=None, mask_as_black=True, mode="mosaic"):
    # Colab環境だとメモリが足りなくなるおそれがあるので、1周目で合計パッチ数を計算し、2周目で格納する 
    train_cnt, test_cnt = 0, 0
    for X, y, z, d in generate_patches(original_patchsize,
                   output_size=output_size, mask_as_black=mask_as_black, mode=mode):
        if d["dataset"]["split"] == "train":
            train_cnt += 1
        else:
            test_cnt += 1

    if output_size is None:
        output_size = original_patchsize

    train = {"image": np.zeros((train_cnt, output_size, output_size, 3), np.uint8),
             "mask": np.zeros((train_cnt, output_size, output_size, 1), np.uint8),
             "mosaic": np.zeros((train_cnt, output_size, output_size, 3), np.uint8),
             "json" : []}
    test  = {"image": np.zeros((test_cnt, output_size, output_size, 3), np.uint8),
             "mask": np.zeros((test_cnt, output_size, output_size, 1), np.uint8),
             "mosaic": np.zeros((test_cnt, output_size, output_size, 3), np.uint8),
             "json": []}

    train_cnt, test_cnt = 0, 0
    for image, mosaic, mask, data in generate_patches(original_patchsize,
                    output_size=output_size, mask_as_black=mask_as_black, mode=mode):
        if data["dataset"]["split"] == "train":
            train["image"][train_cnt] = image
            train["mosaic"][train_cnt] = mosaic
            train["mask"][train_cnt] = mask
            train["json"].append(data)
            train_cnt += 1
        else:
            test["image"][test_cnt] = image
            test["mosaic"][test_cnt] = mosaic
            test["mask"][test_cnt] = mask
            test["json"].append(data)
            test_cnt += 1

    train["json"] = np.array(train["json"])
    test["json"] = np.array(test["json"])

    for d in [train, test]:
        print(d["image"].shape, d["mosaic"].shape, d["mask"].shape, d["json"].shape)

    # (7530, 512, 512, 3) (7530, 512, 512, 3) (7530, 512, 512, 1) (7530,)
    # (1299, 512, 512, 3) (1299, 512, 512, 3) (1299, 512, 512, 1) (1299,)

    # pickleはファイルへの書き込み時に2倍近いメモリオーバーヘッドがあるので、大きいオブジェクトには良くない
    np.savez_compressed("train.npz", **train)
    np.savez_compressed("test.npz", **test)

if __name__ == "__main__":
    pack_data_to_pickle(512, output_size=256, mask_as_black=True, mode="mosaic")