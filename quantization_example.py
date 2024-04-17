{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\ao\\quantization\\fx\\prepare.py:1751: UserWarning: Passing a QConfig dictionary to prepare is deprecated and will not be supported in a future version. Please pass in a QConfigMapping instead.\n",
      "  warnings.warn(\n",
      "c:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\ao\\nn\\quantized\\reference\\modules\\rnn.py:320: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
      "  torch.tensor(weight_qparams[\"scale\"], dtype=torch.float, device=device))\n",
      "c:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\ao\\nn\\quantized\\reference\\modules\\rnn.py:323: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
      "  torch.tensor(weight_qparams[\"zero_point\"], dtype=torch.int, device=device))\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "from torch import nn\n",
    "from torch.quantization import quantize_fx\n",
    "\n",
    "# toy model\n",
    "m = nn.Sequential(\n",
    "  nn.Conv2d(2, 64, kernel_size=8),\n",
    "  nn.ReLU(),\n",
    "  nn.Flatten(),  # Add Flatten layer here\n",
    "  nn.Linear(64, 10),\n",
    "  nn.LSTM(10, 10)\n",
    ")\n",
    "\n",
    "m.eval()\n",
    "\n",
    "# Define example input\n",
    "example_input = torch.rand(1, 2, 8, 8)  # Adjust the size according to your input\n",
    "\n",
    "# FX MODE\n",
    "qconfig_dict = {\"\": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules\n",
    "model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, example_input)\n",
    "model_quantized = quantize_fx.convert_fx(model_prepared)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 1, 2, 8, 8]",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[5], line 12\u001b[0m\n\u001b[0;32m     10\u001b[0m \u001b[38;5;66;03m# Run the non-quantized model\u001b[39;00m\n\u001b[0;32m     11\u001b[0m start \u001b[38;5;241m=\u001b[39m time\u001b[38;5;241m.\u001b[39mtime()\n\u001b[1;32m---> 12\u001b[0m output_non_quantized \u001b[38;5;241m=\u001b[39m \u001b[43mm\u001b[49m\u001b[43m(\u001b[49m\u001b[43minput_data_lstm\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     13\u001b[0m end \u001b[38;5;241m=\u001b[39m time\u001b[38;5;241m.\u001b[39mtime()\n\u001b[0;32m     14\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNon-quantized model output: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00moutput_non_quantized\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32mc:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1511\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1509\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1510\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1511\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1520\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1515\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1516\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1517\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1518\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1519\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1520\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1522\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m   1523\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\nn\\modules\\container.py:217\u001b[0m, in \u001b[0;36mSequential.forward\u001b[1;34m(self, input)\u001b[0m\n\u001b[0;32m    215\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mforward\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;28minput\u001b[39m):\n\u001b[0;32m    216\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m module \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m:\n\u001b[1;32m--> 217\u001b[0m         \u001b[38;5;28minput\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[43mmodule\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43minput\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m    218\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28minput\u001b[39m\n",
      "File \u001b[1;32mc:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1511\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1509\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1510\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1511\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1520\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1515\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1516\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1517\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1518\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1519\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1520\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1522\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m   1523\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\nn\\modules\\conv.py:460\u001b[0m, in \u001b[0;36mConv2d.forward\u001b[1;34m(self, input)\u001b[0m\n\u001b[0;32m    459\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mforward\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;28minput\u001b[39m: Tensor) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Tensor:\n\u001b[1;32m--> 460\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_conv_forward\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43minput\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mweight\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mbias\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32mc:\\Users\\Irfan Q\\miniconda3\\envs\\bruh\\Lib\\site-packages\\torch\\nn\\modules\\conv.py:456\u001b[0m, in \u001b[0;36mConv2d._conv_forward\u001b[1;34m(self, input, weight, bias)\u001b[0m\n\u001b[0;32m    452\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mpadding_mode \u001b[38;5;241m!=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mzeros\u001b[39m\u001b[38;5;124m'\u001b[39m:\n\u001b[0;32m    453\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m F\u001b[38;5;241m.\u001b[39mconv2d(F\u001b[38;5;241m.\u001b[39mpad(\u001b[38;5;28minput\u001b[39m, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_reversed_padding_repeated_twice, mode\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mpadding_mode),\n\u001b[0;32m    454\u001b[0m                     weight, bias, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mstride,\n\u001b[0;32m    455\u001b[0m                     _pair(\u001b[38;5;241m0\u001b[39m), \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdilation, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mgroups)\n\u001b[1;32m--> 456\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mF\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mconv2d\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43minput\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mweight\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mbias\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstride\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    457\u001b[0m \u001b[43m                \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mpadding\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdilation\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mgroups\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[1;31mRuntimeError\u001b[0m: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 1, 2, 8, 8]"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import time\n",
    "\n",
    "# Define your input\n",
    "input_data = torch.rand(1, 2, 8, 8)  # Adjust the size according to your input\n",
    "\n",
    "# Add an extra dimension for LSTM\n",
    "input_data_lstm = input_data.unsqueeze(0)\n",
    "\n",
    "# Run the non-quantized model\n",
    "start = time.time()\n",
    "output_non_quantized = m(input_data_lstm)\n",
    "end = time.time()\n",
    "print(f\"Non-quantized model output: {output_non_quantized}\")\n",
    "print(f\"Non-quantized model execution time: {end - start} seconds\")\n",
    "\n",
    "# Run the quantized model\n",
    "start = time.time()\n",
    "output_quantized = model_quantized(input_data_lstm)\n",
    "end = time.time()\n",
    "print(f\"Quantized model output: {output_quantized}\")\n",
    "print(f\"Quantized model execution time: {end - start} seconds\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "bruh",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
