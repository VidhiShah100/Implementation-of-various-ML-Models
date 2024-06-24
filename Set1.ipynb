{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "##**BITS F464 - Semester 1 - MACHINE LEARNING**\n",
        "--------------------------------------------------------------------------------\n",
        "\n",
        "**ASSIGNMENT 1 - LINEAR MODELS FOR REGRESSION AND CLASSIFICATION**\n",
        "--------------------------------------------------------------------------------\n",
        "***Team number:***\n",
        "6\n",
        "---\n",
        "(*In Title case, separated with commas*)\n",
        "***Full names of all students in the team:***\n",
        "Vidhi Shah, Isha Pargaonkar, Vipanchi Dixit, Shreenidhi Ramaswamy, Shramana Ghosh\n",
        "\n",
        "---\n",
        "(*Separated by commas*)\n",
        "***Id number of all students in the team:***\n",
        "2021A3PS2645H, 2021A3PS2803H, 2021A3PS2983H, 2021A3PS0946H, 2021A7PS1834H"
      ],
      "metadata": {
        "id": "Vj_r89FzT41w"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "This assignment aims to identify the differences between three sets of Machine Learning models."
      ],
      "metadata": {
        "id": "Z_duS0Zn17c3"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **_1. Dataset Generation_**"
      ],
      "metadata": {
        "id": "yT-dTtra2h2n"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "You are given a sample Diabetes dataset. Using this, please develop your own dataset consisting of 500 records. You can use the given code to generate your own dataset. Submit the generated dataset as a .csv file along with your python notebook."
      ],
      "metadata": {
        "id": "2UsOVUj22wrz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install sdv\n",
        "!pip install urllib3==1.26"
      ],
      "metadata": {
        "id": "N8uONwSjNSc-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "23ce2c4d-6948-47ea-d367-dc8dddfd0249"
      },
      "execution_count": 719,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: sdv in /usr/local/lib/python3.10/dist-packages (1.4.0)\n",
            "Requirement already satisfied: boto3<2,>=1.15.0 in /usr/local/lib/python3.10/dist-packages (from sdv) (1.28.57)\n",
            "Requirement already satisfied: botocore<2,>=1.18 in /usr/local/lib/python3.10/dist-packages (from sdv) (1.31.57)\n",
            "Requirement already satisfied: cloudpickle<3.0,>=2.1.0 in /usr/local/lib/python3.10/dist-packages (from sdv) (2.2.1)\n",
            "Requirement already satisfied: Faker<15,>=10 in /usr/local/lib/python3.10/dist-packages (from sdv) (14.2.1)\n",
            "Requirement already satisfied: graphviz<1,>=0.13.2 in /usr/local/lib/python3.10/dist-packages (from sdv) (0.20.1)\n",
            "Requirement already satisfied: tqdm<5,>=4.15 in /usr/local/lib/python3.10/dist-packages (from sdv) (4.66.1)\n",
            "Requirement already satisfied: copulas<0.10,>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from sdv) (0.9.1)\n",
            "Requirement already satisfied: ctgan<0.8,>=0.7.4 in /usr/local/lib/python3.10/dist-packages (from sdv) (0.7.4)\n",
            "Requirement already satisfied: deepecho<0.5,>=0.4.2 in /usr/local/lib/python3.10/dist-packages (from sdv) (0.4.2)\n",
            "Requirement already satisfied: rdt<2,>=1.7.0 in /usr/local/lib/python3.10/dist-packages (from sdv) (1.7.0)\n",
            "Requirement already satisfied: sdmetrics<0.12,>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from sdv) (0.11.1)\n",
            "Requirement already satisfied: numpy<1.25.0,>=1.23.3 in /usr/local/lib/python3.10/dist-packages (from sdv) (1.23.5)\n",
            "Requirement already satisfied: pandas>=1.3.4 in /usr/local/lib/python3.10/dist-packages (from sdv) (1.5.3)\n",
            "Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from boto3<2,>=1.15.0->sdv) (1.0.1)\n",
            "Requirement already satisfied: s3transfer<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from boto3<2,>=1.15.0->sdv) (0.7.0)\n",
            "Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /usr/local/lib/python3.10/dist-packages (from botocore<2,>=1.18->sdv) (2.8.2)\n",
            "Requirement already satisfied: urllib3<1.27,>=1.25.4 in /usr/local/lib/python3.10/dist-packages (from botocore<2,>=1.18->sdv) (1.26.0)\n",
            "Requirement already satisfied: matplotlib<4,>=3.6.0 in /usr/local/lib/python3.10/dist-packages (from copulas<0.10,>=0.9.0->sdv) (3.7.1)\n",
            "Requirement already satisfied: scipy<2,>=1.9.2 in /usr/local/lib/python3.10/dist-packages (from copulas<0.10,>=0.9.0->sdv) (1.11.2)\n",
            "Requirement already satisfied: scikit-learn<2,>=1.1.3 in /usr/local/lib/python3.10/dist-packages (from ctgan<0.8,>=0.7.4->sdv) (1.2.2)\n",
            "Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from ctgan<0.8,>=0.7.4->sdv) (2.0.1+cu118)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.3.4->sdv) (2023.3.post1)\n",
            "Requirement already satisfied: psutil<6,>=5.7 in /usr/local/lib/python3.10/dist-packages (from rdt<2,>=1.7.0->sdv) (5.9.5)\n",
            "Requirement already satisfied: plotly<6,>=5.10.0 in /usr/local/lib/python3.10/dist-packages (from sdmetrics<0.12,>=0.11.0->sdv) (5.15.0)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4,>=3.6.0->copulas<0.10,>=0.9.0->sdv) (1.1.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4,>=3.6.0->copulas<0.10,>=0.9.0->sdv) (0.11.0)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4,>=3.6.0->copulas<0.10,>=0.9.0->sdv) (4.42.1)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4,>=3.6.0->copulas<0.10,>=0.9.0->sdv) (1.4.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4,>=3.6.0->copulas<0.10,>=0.9.0->sdv) (23.1)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4,>=3.6.0->copulas<0.10,>=0.9.0->sdv) (9.4.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<4,>=3.6.0->copulas<0.10,>=0.9.0->sdv) (3.1.1)\n",
            "Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from plotly<6,>=5.10.0->sdmetrics<0.12,>=0.11.0->sdv) (8.2.3)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil<3.0.0,>=2.1->botocore<2,>=1.18->sdv) (1.16.0)\n",
            "Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn<2,>=1.1.3->ctgan<0.8,>=0.7.4->sdv) (1.3.2)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn<2,>=1.1.3->ctgan<0.8,>=0.7.4->sdv) (3.2.0)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (3.12.2)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (4.5.0)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (3.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (3.1.2)\n",
            "Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (2.0.0)\n",
            "Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (3.27.4.1)\n",
            "Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (16.0.6)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (2.1.3)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.11.0->ctgan<0.8,>=0.7.4->sdv) (1.3.0)\n",
            "Requirement already satisfied: urllib3==1.26 in /usr/local/lib/python3.10/dist-packages (1.26.0)\n",
            "\u001b[31mERROR: Operation cancelled by user\u001b[0m\u001b[31m\n",
            "\u001b[0m"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sdv.datasets.local import load_csvs\n",
        "\n",
        "datasets = load_csvs(folder_name='/content/')\n",
        "df = datasets['Diabetes_dataset']"
      ],
      "metadata": {
        "id": "OadtHNAGhL2H",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3e02f99b-cc6a-4d93-9a8d-7f3aeb7bce60"
      },
      "execution_count": 720,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sdv/datasets/local.py:31: UserWarning: Ignoring incompatible files ['my_synthesizer.pkl'] in folder '/content/'.\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sdv.metadata import SingleTableMetadata\n",
        "metadata = SingleTableMetadata()\n",
        "metadata.detect_from_csv(filepath='/content/Diabetes_dataset.csv')"
      ],
      "metadata": {
        "id": "FvvHKTAoh466"
      },
      "execution_count": 721,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "metadata.visualize()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 224
        },
        "id": "W1hIyGlphYSx",
        "outputId": "b9b2a789-3d90-4d45-c110-435e4fbd400a"
      },
      "execution_count": 722,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "image/svg+xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Generated by graphviz version 2.43.0 (0)\n -->\n<!-- Title: Metadata Pages: 1 -->\n<svg width=\"233pt\" height=\"152pt\"\n viewBox=\"0.00 0.00 233.00 152.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 148)\">\n<title>Metadata</title>\n<polygon fill=\"white\" stroke=\"transparent\" points=\"-4,4 -4,-148 229,-148 229,4 -4,4\"/>\n<g id=\"node1\" class=\"node\">\n<title></title>\n<path fill=\"#ffec8b\" stroke=\"black\" d=\"M12,-0.5C12,-0.5 213,-0.5 213,-0.5 219,-0.5 225,-6.5 225,-12.5 225,-12.5 225,-131.5 225,-131.5 225,-137.5 219,-143.5 213,-143.5 213,-143.5 12,-143.5 12,-143.5 6,-143.5 0,-137.5 0,-131.5 0,-131.5 0,-12.5 0,-12.5 0,-6.5 6,-0.5 12,-0.5\"/>\n<text text-anchor=\"start\" x=\"8\" y=\"-128.3\" font-family=\"Times,serif\" font-size=\"14.00\">Pregnancies : numerical</text>\n<text text-anchor=\"start\" x=\"8\" y=\"-113.3\" font-family=\"Times,serif\" font-size=\"14.00\">Glucose : numerical</text>\n<text text-anchor=\"start\" x=\"8\" y=\"-98.3\" font-family=\"Times,serif\" font-size=\"14.00\">BloodPressure : numerical</text>\n<text text-anchor=\"start\" x=\"8\" y=\"-83.3\" font-family=\"Times,serif\" font-size=\"14.00\">SkinThickness : numerical</text>\n<text text-anchor=\"start\" x=\"8\" y=\"-68.3\" font-family=\"Times,serif\" font-size=\"14.00\">Insulin : numerical</text>\n<text text-anchor=\"start\" x=\"8\" y=\"-53.3\" font-family=\"Times,serif\" font-size=\"14.00\">BMI : numerical</text>\n<text text-anchor=\"start\" x=\"8\" y=\"-38.3\" font-family=\"Times,serif\" font-size=\"14.00\">DiabetesPedigreeFunction : numerical</text>\n<text text-anchor=\"start\" x=\"8\" y=\"-23.3\" font-family=\"Times,serif\" font-size=\"14.00\">Age : numerical</text>\n<text text-anchor=\"start\" x=\"8\" y=\"-8.3\" font-family=\"Times,serif\" font-size=\"14.00\">Outcome : numerical</text>\n</g>\n</g>\n</svg>\n",
            "text/plain": [
              "<graphviz.graphs.Digraph at 0x78fe1d857b50>"
            ]
          },
          "metadata": {},
          "execution_count": 722
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sdv.lite import SingleTablePreset\n",
        "\n",
        "synthesizer = SingleTablePreset(\n",
        "    metadata,\n",
        "    name='FAST_ML'\n",
        ")"
      ],
      "metadata": {
        "id": "A-V1GDqoiBfs"
      },
      "execution_count": 723,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "synthesizer.fit(\n",
        "    data=datasets['Diabetes_dataset']\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 369
        },
        "id": "C7LQocMYiEMp",
        "outputId": "e774461a-3479-4b1a-e11d-e6f47fba94d8"
      },
      "execution_count": 724,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ContextualVersionConflict",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mContextualVersionConflict\u001b[0m                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-724-9f457607bfb8>\u001b[0m in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m synthesizer.fit(\n\u001b[0m\u001b[1;32m      2\u001b[0m     \u001b[0mdata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdatasets\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'Diabetes_dataset'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m )\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sdv/lite/single_table.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, data)\u001b[0m\n\u001b[1;32m     78\u001b[0m                 \u001b[0mData\u001b[0m \u001b[0mto\u001b[0m \u001b[0mfit\u001b[0m \u001b[0mthe\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0mto\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     79\u001b[0m         \"\"\"\n\u001b[0;32m---> 80\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_synthesizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     81\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     82\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0msample\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_rows\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmax_tries_per_batch\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m100\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moutput_file_path\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sdv/single_table/base.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, data)\u001b[0m\n\u001b[1;32m    376\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_random_state_set\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    377\u001b[0m         \u001b[0mprocessed_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_preprocess\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 378\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit_processed_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mprocessed_data\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    379\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    380\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0msave\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfilepath\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sdv/single_table/base.py\u001b[0m in \u001b[0;36mfit_processed_data\u001b[0;34m(self, processed_data)\u001b[0m\n\u001b[1;32m    363\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_fitted\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    364\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_fitted_date\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdatetime\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdatetime\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtoday\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstrftime\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'%Y-%m-%d'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 365\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_fitted_sdv_version\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpkg_resources\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_distribution\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'sdv'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mversion\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    366\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    367\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py\u001b[0m in \u001b[0;36mget_distribution\u001b[0;34m(dist)\u001b[0m\n\u001b[1;32m    524\u001b[0m         \u001b[0mdist\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mRequirement\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparse\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdist\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    525\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdist\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mRequirement\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 526\u001b[0;31m         \u001b[0mdist\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mget_provider\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdist\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    527\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdist\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mDistribution\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    528\u001b[0m         \u001b[0;32mraise\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Expected string, Requirement, or Distribution\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdist\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py\u001b[0m in \u001b[0;36mget_provider\u001b[0;34m(moduleOrReq)\u001b[0m\n\u001b[1;32m    396\u001b[0m     \u001b[0;34m\"\"\"Return an IResourceProvider for the named module or requirement\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    397\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmoduleOrReq\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mRequirement\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 398\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mworking_set\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfind\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmoduleOrReq\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0mrequire\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmoduleOrReq\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    399\u001b[0m     \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    400\u001b[0m         \u001b[0mmodule\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmodules\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mmoduleOrReq\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py\u001b[0m in \u001b[0;36mrequire\u001b[0;34m(self, *requirements)\u001b[0m\n\u001b[1;32m    964\u001b[0m         \u001b[0mincluded\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0meven\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mthey\u001b[0m \u001b[0mwere\u001b[0m \u001b[0malready\u001b[0m \u001b[0mactivated\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mthis\u001b[0m \u001b[0mworking\u001b[0m \u001b[0mset\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    965\u001b[0m         \"\"\"\n\u001b[0;32m--> 966\u001b[0;31m         \u001b[0mneeded\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresolve\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mparse_requirements\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrequirements\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    967\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    968\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mdist\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mneeded\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py\u001b[0m in \u001b[0;36mresolve\u001b[0;34m(self, requirements, env, installer, replace_conflicting, extras)\u001b[0m\n\u001b[1;32m    825\u001b[0m                 \u001b[0;32mcontinue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    826\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 827\u001b[0;31m             dist = self._resolve_dist(\n\u001b[0m\u001b[1;32m    828\u001b[0m                 \u001b[0mreq\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbest\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreplace_conflicting\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0menv\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minstaller\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrequired_by\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mto_activate\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    829\u001b[0m             )\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py\u001b[0m in \u001b[0;36m_resolve_dist\u001b[0;34m(self, req, best, replace_conflicting, env, installer, required_by, to_activate)\u001b[0m\n\u001b[1;32m    871\u001b[0m             \u001b[0;31m# Oops, the \"best\" so far conflicts with a dependency\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    872\u001b[0m             \u001b[0mdependent_req\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrequired_by\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mreq\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 873\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mVersionConflict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdist\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreq\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwith_context\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdependent_req\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    874\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mdist\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    875\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mContextualVersionConflict\u001b[0m: (urllib3 2.0.4 (/usr/local/lib/python3.10/dist-packages), Requirement.parse('urllib3<1.27,>=1.25.4'), {'botocore'})"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "id": "Qsf7BCqyiLL-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "synthetic_data = synthesizer.sample(\n",
        "    num_rows=500\n",
        ")\n",
        "synthetic_data.head()"
      ],
      "metadata": {
        "id": "-ndPzUApiPkC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sdv.evaluation.single_table import evaluate_quality\n",
        "\n",
        "quality_report = evaluate_quality(\n",
        "    df,\n",
        "    synthetic_data,\n",
        "    metadata\n",
        ")"
      ],
      "metadata": {
        "id": "0DmWf7B4iS4-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "synthesizer.save('my_synthesizer.pkl')\n",
        "\n",
        "synthesizer = SingleTablePreset.load('my_synthesizer.pkl')"
      ],
      "metadata": {
        "id": "U1BVzS3tiXhk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "synthetic_data"
      ],
      "metadata": {
        "id": "MqvIPN3aia0D"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "synthesizer\n",
        "synthetic_data.to_csv('Diabetes_Dataset.csv')"
      ],
      "metadata": {
        "id": "tbJ0ctdwidyN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "\n",
        "\n",
        "# ***2. Preprocess and perform exploratory data analysis of the dataset obtained***"
      ],
      "metadata": {
        "id": "rDu7bwbNRhaK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from matplotlib import pyplot as plt\n",
        "%matplotlib inline\n",
        "import random\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.metrics import f1_score\n",
        "from sklearn import metrics\n"
      ],
      "metadata": {
        "id": "oAd4cNwERr90"
      },
      "execution_count": 725,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "gdf=pd.read_csv('Diabetes_Dataset.csv')\n",
        "gdf.head()"
      ],
      "metadata": {
        "id": "kBTg0pQuXEt-",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "ccf92285-2ffe-484a-99c9-da89e3a9d9be"
      },
      "execution_count": 726,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Unnamed: 0  Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin  \\\n",
              "0           0            7      149             96             19       10   \n",
              "1           1            0      151             44              6      105   \n",
              "2           2            3      169             57             24      240   \n",
              "3           3            4       86             61             35        0   \n",
              "4           4            6       75             62             31       77   \n",
              "\n",
              "         BMI  DiabetesPedigreeFunction  Age  Outcome  \n",
              "0  38.387409                  0.561331   40        1  \n",
              "1  26.125923                  0.463959   27        1  \n",
              "2  33.224573                  0.541364   36        0  \n",
              "3  32.918264                  0.526311   39        0  \n",
              "4  37.453830                  0.178734   21        0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-2b0b51c6-71a3-41f5-988c-70d48e304cce\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>Pregnancies</th>\n",
              "      <th>Glucose</th>\n",
              "      <th>BloodPressure</th>\n",
              "      <th>SkinThickness</th>\n",
              "      <th>Insulin</th>\n",
              "      <th>BMI</th>\n",
              "      <th>DiabetesPedigreeFunction</th>\n",
              "      <th>Age</th>\n",
              "      <th>Outcome</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>7</td>\n",
              "      <td>149</td>\n",
              "      <td>96</td>\n",
              "      <td>19</td>\n",
              "      <td>10</td>\n",
              "      <td>38.387409</td>\n",
              "      <td>0.561331</td>\n",
              "      <td>40</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>151</td>\n",
              "      <td>44</td>\n",
              "      <td>6</td>\n",
              "      <td>105</td>\n",
              "      <td>26.125923</td>\n",
              "      <td>0.463959</td>\n",
              "      <td>27</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>3</td>\n",
              "      <td>169</td>\n",
              "      <td>57</td>\n",
              "      <td>24</td>\n",
              "      <td>240</td>\n",
              "      <td>33.224573</td>\n",
              "      <td>0.541364</td>\n",
              "      <td>36</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>86</td>\n",
              "      <td>61</td>\n",
              "      <td>35</td>\n",
              "      <td>0</td>\n",
              "      <td>32.918264</td>\n",
              "      <td>0.526311</td>\n",
              "      <td>39</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>6</td>\n",
              "      <td>75</td>\n",
              "      <td>62</td>\n",
              "      <td>31</td>\n",
              "      <td>77</td>\n",
              "      <td>37.453830</td>\n",
              "      <td>0.178734</td>\n",
              "      <td>21</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2b0b51c6-71a3-41f5-988c-70d48e304cce')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-2b0b51c6-71a3-41f5-988c-70d48e304cce button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-2b0b51c6-71a3-41f5-988c-70d48e304cce');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-a47f2ccb-c5f7-4a45-a7a1-7fb067cd7678\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-a47f2ccb-c5f7-4a45-a7a1-7fb067cd7678')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-a47f2ccb-c5f7-4a45-a7a1-7fb067cd7678 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 726
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "gdf.isna().sum()\n",
        "# there are no Nan values in the dataset generated"
      ],
      "metadata": {
        "id": "lNrGgwM_kp59",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0588249c-6443-4cf9-b598-205ccc5b2ed6"
      },
      "execution_count": 727,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Unnamed: 0                  0\n",
              "Pregnancies                 0\n",
              "Glucose                     0\n",
              "BloodPressure               0\n",
              "SkinThickness               0\n",
              "Insulin                     0\n",
              "BMI                         0\n",
              "DiabetesPedigreeFunction    0\n",
              "Age                         0\n",
              "Outcome                     0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 727
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "gdf.isnull().sum()\n",
        "# there are no null values in the dataset generated"
      ],
      "metadata": {
        "id": "z3ThopRWkt1g",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5c1f8adb-67b6-4c7d-b2c6-51478998b613"
      },
      "execution_count": 728,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Unnamed: 0                  0\n",
              "Pregnancies                 0\n",
              "Glucose                     0\n",
              "BloodPressure               0\n",
              "SkinThickness               0\n",
              "Insulin                     0\n",
              "BMI                         0\n",
              "DiabetesPedigreeFunction    0\n",
              "Age                         0\n",
              "Outcome                     0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 728
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "gdf.info()"
      ],
      "metadata": {
        "id": "MeiKCxapk5ZM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9ddae4de-6763-467d-bff3-1b125e6787d0"
      },
      "execution_count": 729,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 500 entries, 0 to 499\n",
            "Data columns (total 10 columns):\n",
            " #   Column                    Non-Null Count  Dtype  \n",
            "---  ------                    --------------  -----  \n",
            " 0   Unnamed: 0                500 non-null    int64  \n",
            " 1   Pregnancies               500 non-null    int64  \n",
            " 2   Glucose                   500 non-null    int64  \n",
            " 3   BloodPressure             500 non-null    int64  \n",
            " 4   SkinThickness             500 non-null    int64  \n",
            " 5   Insulin                   500 non-null    int64  \n",
            " 6   BMI                       500 non-null    float64\n",
            " 7   DiabetesPedigreeFunction  500 non-null    float64\n",
            " 8   Age                       500 non-null    int64  \n",
            " 9   Outcome                   500 non-null    int64  \n",
            "dtypes: float64(2), int64(8)\n",
            "memory usage: 39.2 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "gdf.describe()"
      ],
      "metadata": {
        "id": "UXVbx0F3kyOa",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "outputId": "f948dca7-7969-4017-d47b-03b6ccc5f26d"
      },
      "execution_count": 730,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       Unnamed: 0  Pregnancies     Glucose  BloodPressure  SkinThickness  \\\n",
              "count  500.000000   500.000000  500.000000     500.000000     500.000000   \n",
              "mean   249.500000     4.006000  120.454000      71.080000      21.778000   \n",
              "std    144.481833     3.052967   33.702754      18.964517      13.916467   \n",
              "min      0.000000     0.000000   14.000000      22.000000       0.000000   \n",
              "25%    124.750000     2.000000   98.000000      58.000000      11.000000   \n",
              "50%    249.500000     4.000000  120.500000      71.000000      22.000000   \n",
              "75%    374.250000     6.000000  145.000000      85.000000      32.000000   \n",
              "max    499.000000    17.000000  199.000000     122.000000      55.000000   \n",
              "\n",
              "          Insulin         BMI  DiabetesPedigreeFunction       Age     Outcome  \n",
              "count  500.000000  500.000000                500.000000  500.0000  500.000000  \n",
              "mean    97.290000   32.518053                  0.511352   34.4500    0.330000  \n",
              "std     92.202932    7.887027                  0.299594   10.0826    0.470684  \n",
              "min      0.000000   10.903340                  0.078000   21.0000    0.000000  \n",
              "25%      1.000000   27.161187                  0.255483   27.0000    0.000000  \n",
              "50%     78.000000   32.593184                  0.508918   33.0000    0.000000  \n",
              "75%    165.000000   37.500736                  0.735819   41.0000    1.000000  \n",
              "max    378.000000   59.937412                  1.413874   73.0000    1.000000  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-3e7b03c2-f10d-4ef2-bae2-055318ad7140\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>Pregnancies</th>\n",
              "      <th>Glucose</th>\n",
              "      <th>BloodPressure</th>\n",
              "      <th>SkinThickness</th>\n",
              "      <th>Insulin</th>\n",
              "      <th>BMI</th>\n",
              "      <th>DiabetesPedigreeFunction</th>\n",
              "      <th>Age</th>\n",
              "      <th>Outcome</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>500.000000</td>\n",
              "      <td>500.000000</td>\n",
              "      <td>500.000000</td>\n",
              "      <td>500.000000</td>\n",
              "      <td>500.000000</td>\n",
              "      <td>500.000000</td>\n",
              "      <td>500.000000</td>\n",
              "      <td>500.000000</td>\n",
              "      <td>500.0000</td>\n",
              "      <td>500.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>249.500000</td>\n",
              "      <td>4.006000</td>\n",
              "      <td>120.454000</td>\n",
              "      <td>71.080000</td>\n",
              "      <td>21.778000</td>\n",
              "      <td>97.290000</td>\n",
              "      <td>32.518053</td>\n",
              "      <td>0.511352</td>\n",
              "      <td>34.4500</td>\n",
              "      <td>0.330000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>144.481833</td>\n",
              "      <td>3.052967</td>\n",
              "      <td>33.702754</td>\n",
              "      <td>18.964517</td>\n",
              "      <td>13.916467</td>\n",
              "      <td>92.202932</td>\n",
              "      <td>7.887027</td>\n",
              "      <td>0.299594</td>\n",
              "      <td>10.0826</td>\n",
              "      <td>0.470684</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>14.000000</td>\n",
              "      <td>22.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>10.903340</td>\n",
              "      <td>0.078000</td>\n",
              "      <td>21.0000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>124.750000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>98.000000</td>\n",
              "      <td>58.000000</td>\n",
              "      <td>11.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>27.161187</td>\n",
              "      <td>0.255483</td>\n",
              "      <td>27.0000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>249.500000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>120.500000</td>\n",
              "      <td>71.000000</td>\n",
              "      <td>22.000000</td>\n",
              "      <td>78.000000</td>\n",
              "      <td>32.593184</td>\n",
              "      <td>0.508918</td>\n",
              "      <td>33.0000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>374.250000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>145.000000</td>\n",
              "      <td>85.000000</td>\n",
              "      <td>32.000000</td>\n",
              "      <td>165.000000</td>\n",
              "      <td>37.500736</td>\n",
              "      <td>0.735819</td>\n",
              "      <td>41.0000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>499.000000</td>\n",
              "      <td>17.000000</td>\n",
              "      <td>199.000000</td>\n",
              "      <td>122.000000</td>\n",
              "      <td>55.000000</td>\n",
              "      <td>378.000000</td>\n",
              "      <td>59.937412</td>\n",
              "      <td>1.413874</td>\n",
              "      <td>73.0000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-3e7b03c2-f10d-4ef2-bae2-055318ad7140')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-3e7b03c2-f10d-4ef2-bae2-055318ad7140 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-3e7b03c2-f10d-4ef2-bae2-055318ad7140');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-b2677f77-da6b-4a67-91ca-8409abc29ff0\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-b2677f77-da6b-4a67-91ca-8409abc29ff0')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-b2677f77-da6b-4a67-91ca-8409abc29ff0 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 730
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "In the code block below, we have replaced the vaules of SkinThickness which were 0 in the original dataset by their mean as skin thickness cannot be 0."
      ],
      "metadata": {
        "id": "OUzBmQmjV7Vv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "column=gdf['SkinThickness']\n",
        "column.replace(to_replace = 0, value = column.mean(), inplace=True)\n",
        "gdf['SkinThickness']"
      ],
      "metadata": {
        "id": "aLV2gPwBk-oc",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d4e91d4a-9ab4-4f3b-c0ff-e15b764aa691"
      },
      "execution_count": 731,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0      19.0\n",
              "1       6.0\n",
              "2      24.0\n",
              "3      35.0\n",
              "4      31.0\n",
              "       ... \n",
              "495    46.0\n",
              "496    28.0\n",
              "497    38.0\n",
              "498    49.0\n",
              "499    22.0\n",
              "Name: SkinThickness, Length: 500, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 731
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dataset=(gdf-gdf.min())/(gdf.max()-gdf.min())\n",
        "X=dataset.drop(['Outcome', 'Unnamed: 0'], axis=1)\n",
        "y=dataset.Outcome\n",
        "dataset2=dataset.drop(['Unnamed: 0'], axis=1) #dataset without the 'Unnamed: 0' column"
      ],
      "metadata": {
        "id": "4doeqpJOlByP"
      },
      "execution_count": 732,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#correlation visualization\n",
        "dataplot = sns.heatmap(dataset2.corr(), cmap=\"YlGnBu\", annot=True)\n",
        "plt.show()\n",
        "dataset2"
      ],
      "metadata": {
        "id": "AT-fULIgYIZ3",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "47f06785-6128-4eab-ae9b-0c381ac78abb"
      },
      "execution_count": 733,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin       BMI  \\\n",
              "0       0.411765  0.729730           0.74       0.333333  0.026455  0.560510   \n",
              "1       0.000000  0.740541           0.22       0.092593  0.277778  0.310449   \n",
              "2       0.176471  0.837838           0.35       0.425926  0.634921  0.455219   \n",
              "3       0.235294  0.389189           0.39       0.629630  0.000000  0.448972   \n",
              "4       0.352941  0.329730           0.40       0.555556  0.203704  0.541470   \n",
              "..           ...       ...            ...            ...       ...       ...   \n",
              "495     0.000000  0.470270           0.65       0.833333  0.785714  0.567543   \n",
              "496     0.235294  0.594595           0.50       0.500000  0.071429  0.462390   \n",
              "497     0.235294  0.902703           0.13       0.685185  0.658730  0.494231   \n",
              "498     0.235294  0.745946           0.59       0.888889  0.436508  0.525994   \n",
              "499     0.058824  0.513514           0.33       0.388889  0.100529  0.591608   \n",
              "\n",
              "     DiabetesPedigreeFunction       Age  Outcome  \n",
              "0                    0.361809  0.365385      1.0  \n",
              "1                    0.288919  0.115385      1.0  \n",
              "2                    0.346862  0.288462      0.0  \n",
              "3                    0.335594  0.346154      0.0  \n",
              "4                    0.075407  0.000000      0.0  \n",
              "..                        ...       ...      ...  \n",
              "495                  0.257074  0.000000      0.0  \n",
              "496                  0.413268  0.557692      0.0  \n",
              "497                  0.267186  0.269231      1.0  \n",
              "498                  0.556111  0.461538      1.0  \n",
              "499                  0.445205  0.000000      0.0  \n",
              "\n",
              "[500 rows x 9 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-ff6c752c-f0d1-44b9-a114-86c3e44fd765\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Pregnancies</th>\n",
              "      <th>Glucose</th>\n",
              "      <th>BloodPressure</th>\n",
              "      <th>SkinThickness</th>\n",
              "      <th>Insulin</th>\n",
              "      <th>BMI</th>\n",
              "      <th>DiabetesPedigreeFunction</th>\n",
              "      <th>Age</th>\n",
              "      <th>Outcome</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.411765</td>\n",
              "      <td>0.729730</td>\n",
              "      <td>0.74</td>\n",
              "      <td>0.333333</td>\n",
              "      <td>0.026455</td>\n",
              "      <td>0.560510</td>\n",
              "      <td>0.361809</td>\n",
              "      <td>0.365385</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.740541</td>\n",
              "      <td>0.22</td>\n",
              "      <td>0.092593</td>\n",
              "      <td>0.277778</td>\n",
              "      <td>0.310449</td>\n",
              "      <td>0.288919</td>\n",
              "      <td>0.115385</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.176471</td>\n",
              "      <td>0.837838</td>\n",
              "      <td>0.35</td>\n",
              "      <td>0.425926</td>\n",
              "      <td>0.634921</td>\n",
              "      <td>0.455219</td>\n",
              "      <td>0.346862</td>\n",
              "      <td>0.288462</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.235294</td>\n",
              "      <td>0.389189</td>\n",
              "      <td>0.39</td>\n",
              "      <td>0.629630</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.448972</td>\n",
              "      <td>0.335594</td>\n",
              "      <td>0.346154</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.352941</td>\n",
              "      <td>0.329730</td>\n",
              "      <td>0.40</td>\n",
              "      <td>0.555556</td>\n",
              "      <td>0.203704</td>\n",
              "      <td>0.541470</td>\n",
              "      <td>0.075407</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>495</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.470270</td>\n",
              "      <td>0.65</td>\n",
              "      <td>0.833333</td>\n",
              "      <td>0.785714</td>\n",
              "      <td>0.567543</td>\n",
              "      <td>0.257074</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>496</th>\n",
              "      <td>0.235294</td>\n",
              "      <td>0.594595</td>\n",
              "      <td>0.50</td>\n",
              "      <td>0.500000</td>\n",
              "      <td>0.071429</td>\n",
              "      <td>0.462390</td>\n",
              "      <td>0.413268</td>\n",
              "      <td>0.557692</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>497</th>\n",
              "      <td>0.235294</td>\n",
              "      <td>0.902703</td>\n",
              "      <td>0.13</td>\n",
              "      <td>0.685185</td>\n",
              "      <td>0.658730</td>\n",
              "      <td>0.494231</td>\n",
              "      <td>0.267186</td>\n",
              "      <td>0.269231</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>498</th>\n",
              "      <td>0.235294</td>\n",
              "      <td>0.745946</td>\n",
              "      <td>0.59</td>\n",
              "      <td>0.888889</td>\n",
              "      <td>0.436508</td>\n",
              "      <td>0.525994</td>\n",
              "      <td>0.556111</td>\n",
              "      <td>0.461538</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>499</th>\n",
              "      <td>0.058824</td>\n",
              "      <td>0.513514</td>\n",
              "      <td>0.33</td>\n",
              "      <td>0.388889</td>\n",
              "      <td>0.100529</td>\n",
              "      <td>0.591608</td>\n",
              "      <td>0.445205</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>500 rows × 9 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ff6c752c-f0d1-44b9-a114-86c3e44fd765')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-ff6c752c-f0d1-44b9-a114-86c3e44fd765 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-ff6c752c-f0d1-44b9-a114-86c3e44fd765');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-0979d392-86e6-4652-b404-fdf1a7a41ee6\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-0979d392-86e6-4652-b404-fdf1a7a41ee6')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-0979d392-86e6-4652-b404-fdf1a7a41ee6 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 733
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "gdf['Outcome'].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1FrnrjeiTU9w",
        "outputId": "fd8c8b6a-d04a-4980-ee52-d541ebb10649"
      },
      "execution_count": 734,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    335\n",
              "1    165\n",
              "Name: Outcome, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 734
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "As we can see from the preprocessed data, the data samples for 1 are almost twice as less than the samples for 0. Hence, this irregularity could lower accuracy of the models."
      ],
      "metadata": {
        "id": "jvjQjikNVXHd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=9)"
      ],
      "metadata": {
        "id": "JWmYYHf8ebvf"
      },
      "execution_count": 735,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# ***3. Comparison of Stochastic Gradient Descent and Batch Gradient Descent using Linear Regression***\n",
        "\n",
        "## **_Stochastic Gradient Descent_**"
      ],
      "metadata": {
        "id": "y81SQxOrRso5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "Y=np.array(dataset.iloc[0:,9:])\n",
        "\n",
        "df = dataset.drop('Outcome', axis=1)\n",
        "\n",
        "#scaling the data\n",
        "dataset = (df-df.min())/(df.max()-df.min())\n",
        "\n",
        "X=dataset.iloc[0:,1:9]\n",
        "\n",
        "#adding bias column\n",
        "X['D']=1\n",
        "X=np.array(X)\n",
        "\n",
        "#applying train_test_split\n",
        "from sklearn.model_selection import train_test_split\n",
        "x_train,x_test,y_train,y_test=train_test_split(X, Y,train_size=0.8, test_size=0.2,random_state=9)\n",
        "\n",
        "w=np.zeros(9) #initialize the weight vector\n",
        "lr=0.001 #lr=learning rate\n",
        "iter=1000\n",
        "N=len(X) #N= total number of samples\n",
        "n=len(x_train) #n = number of samples in the training dataset\n",
        "sgd_cost_list=[] #list recording the error after each iteration\n",
        "sgd_epoch_list=[] #list recording the iteration count\n",
        "\n",
        "#define cost function\n",
        "def costfn(X,Y,w):\n",
        "  error=0\n",
        "  for i in range(n):\n",
        "    x=X[i]\n",
        "    y=Y[i]\n",
        "    h=(w.T).dot(x) #formulating the hypothesis\n",
        "    error+=(h-y)**2\n",
        "  error = (1/n)*error #computing the error\n",
        "  sgd_cost_list.append(error) #adding error value to the cost list\n",
        "\n",
        "#Stochastic descent\n",
        "def stochastic_des(X,Y,w,lr,iter,n):\n",
        "  for m in range(iter):\n",
        "    sum=np.zeros(9)\n",
        "    j=random.randint(0,n-1) #picking a random sample from the dataset\n",
        "    x=X[j]\n",
        "    y=Y[j]\n",
        "    h=(w.T).dot(x) #computing hypothesis\n",
        "    sum=(h-y)*x #computing derivative of cost function\n",
        "    w=w-(lr*sum)\n",
        "    costfn(X,Y,w)\n",
        "    sgd_epoch_list.append(m)\n",
        "  return w\n",
        "\n",
        "w_new=stochastic_des(x_train, y_train, w,lr,iter,n)\n",
        "\n",
        "test=len(y_test)\n",
        "#y_pred=np.ones(test)\n",
        "y_p=np.arange(test)\n",
        "y_pred=y_p.reshape(test,1)\n",
        "\n",
        "def Linear_Regression(x_test,test):\n",
        "  for i in range(test):\n",
        "    y_pred[i] = np.round(abs((w_new.T).dot(x_test[i])))\n",
        "  return y_pred\n",
        "\n",
        "y_pred=Linear_Regression(x_test,test)\n",
        "mse= mean_squared_error(y_test,y_pred)\n",
        "def accuracy(y_pred,y_test):\n",
        "    return np.sum(y_pred==y_test)/len(y_test)\n",
        "final_accuracy=accuracy(y_pred,y_test)\n",
        "print(f\"mse:\", mse)\n",
        "print(f\"Accuracy : {final_accuracy: .2f}\")\n",
        "print(f\"Weights:\", w_new)"
      ],
      "metadata": {
        "id": "tCDbDv9ekw37",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "19bba032-e537-4b7c-bffc-b212bcbd820b"
      },
      "execution_count": 736,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mse: 0.36\n",
            "Accuracy :  0.64\n",
            "Weights: [0.04293597 0.09538799 0.0524842  0.05182083 0.04422385 0.05847898\n",
            " 0.05159613 0.04094278 0.11307163]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **_Batch Gradient Descent_**"
      ],
      "metadata": {
        "id": "HqIC34nWR-n7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "w=np.zeros(9) #initialize the weight vector\n",
        "lr=0.001 #lr=learning rate\n",
        "iter=1000\n",
        "N=len(X) #N= total number of samples\n",
        "n=len(x_train) #n = number of samples in the training dataset\n",
        "\n",
        "bgd_cost_list=[] #list recording the error after each iteration\n",
        "bgd_epoch_list=[] #list recording the iteration count"
      ],
      "metadata": {
        "id": "-hA5FysjSBzj"
      },
      "execution_count": 737,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def costfn(X,Y,w):\n",
        "  error=0\n",
        "  for i in range(n):\n",
        "     x=X[i]\n",
        "     y=Y[i]\n",
        "     h=(w.T).dot(x) #formulating the hypothesis\n",
        "     error += (h-y)**2\n",
        "  error=(1/n)*error #computing the error\n",
        "  bgd_cost_list.append(error) #adding error value to the cost list\n",
        "  return error\n",
        "\n",
        "#Gradient descent\n",
        "def grad_des(X,Y,w,lr,iter,n):\n",
        "  for m in range(iter):\n",
        "    sum=np.zeros(9)\n",
        "    for i in range(n):\n",
        "      x=X[i]\n",
        "      y=Y[i]\n",
        "      h=(w.T).dot(x) #formulating the hypothesis\n",
        "      sum+=(h-y)*x #computing derivative of cost function\n",
        "\n",
        "    w=w-(lr*sum)\n",
        "    costfn(x_train,y_train,w)\n",
        "    bgd_epoch_list.append(m)\n",
        "  return w\n",
        "\n",
        "w_new=grad_des(x_train, y_train, w,lr,iter,n)\n"
      ],
      "metadata": {
        "id": "jmH1jxwRlmf_"
      },
      "execution_count": 738,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test=len(y_test)\n",
        "y_p=np.arange(test)\n",
        "y_pred=y_p.reshape(test,1)\n",
        "\n",
        "def Linear_Regression(x_test,test):\n",
        "  for i in range(test):\n",
        "    y_pred[i] = np.round(abs((w_new.T).dot(x_test[i])))\n",
        "  return y_pred\n",
        "\n",
        "y_pred=Linear_Regression(x_test,test)\n",
        "mse= mean_squared_error(y_test,y_pred)\n",
        "def accuracy(y_pred,y_test):\n",
        "    return np.sum(y_pred==y_test)/len(y_test)\n",
        "final_accuracy=accuracy(y_pred,y_test)\n",
        "\n",
        "print(f\"mse:\", mse)\n",
        "print(f\"Accuracy : {final_accuracy: .2f}\")\n",
        "print(f\"Weights:\", w_new)"
      ],
      "metadata": {
        "id": "nvMJxJOwlpFs",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f88f6d33-7d14-4f71-f38f-0e4bbd79e1f3"
      },
      "execution_count": 739,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mse: 0.28\n",
            "Accuracy :  0.72\n",
            "Weights: [ 0.53361169  0.9989401  -0.29022008  0.20614123 -0.10371824  0.38984752\n",
            "  0.15398813 -0.12610462 -0.47588348]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **_Insights drawn (plots, markdown explanations)_**"
      ],
      "metadata": {
        "id": "8g7NxS5sSCRv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.title(\"Batch gradient descent plot\")\n",
        "plt.xlabel(\"epoch\")\n",
        "plt.ylabel(\"error\")\n",
        "plt.plot(bgd_epoch_list,bgd_cost_list)"
      ],
      "metadata": {
        "id": "Pw63z3amSSo_",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 490
        },
        "outputId": "045b1cf9-47be-4962-8a53-5b0b750e4bf4"
      },
      "execution_count": 740,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x78fe0dc21ed0>]"
            ]
          },
          "metadata": {},
          "execution_count": 740
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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M187BbXht7h2mqpl+7X9Rb32xma3Nnd9jfdvSt8mbLOhPo2sr6WsOfXXWmuy5mxkYcWjJedURwYkIrJpCoXi6qkyuSshovaEl6EQERER1cGARERERFQHAxIRERFRHQxIRERERHUwIBERERHVwYBEREREVAcDEhEREVEdDEhEREREdTAgEREREdXBgERERERUBwMSERERUR0MSERERER1MCARERER1cGARERERFSHg9wF2CshBADAYDDIXAkRERE1Ve3v7drf441hQGqm0tJSAEBgYKDMlRAREdHNKi0thU6na3S5QtwoQlGDzGYzcnNz4ebmBoVCYbX1GgwGBAYGIjs7G+7u7lZbL9XHY902eJzbBo9z2+GxbhutdZyFECgtLUVAQACUysZnGnEEqZmUSiU6derUaut3d3fn/3hthMe6bfA4tw0e57bDY902WuM4X2/kqBYnaRMRERHVwYBEREREVAcDko3RaDSYP38+NBqN3KXc8nis2waPc9vgcW47PNZtQ+7jzEnaRERERHVwBImIiIioDgYkIiIiojoYkIiIiIjqYEAiIiIiqoMBycasXLkSQUFB0Gq1iIyMxJ49e+Quya4kJSVh4MCBcHNzg6+vL8aMGYNjx45Z9Lly5QpmzpyJDh06wNXVFXFxccjPz7fok5WVhdGjR8PZ2Rm+vr547rnnUF1d3Za7YleWLFkChUKB2bNnS208ztaRk5ODiRMnokOHDnByckJ4eDh++eUXabkQAomJifD394eTkxNiYmJw4sQJi3VcvHgREyZMgLu7Ozw8PPDoo4+irKysrXfFZplMJsybNw/BwcFwcnJCSEgIFi1aZPGsLh7n5vnhhx9w7733IiAgAAqFAps2bbJYbq3j+ttvv2HYsGHQarUIDAzE0qVLW168IJuxYcMGoVarxXvvvScOHTokpk2bJjw8PER+fr7cpdmN2NhYsWbNGnHw4EGRmZkp7rnnHtG5c2dRVlYm9Xn88cdFYGCgSE1NFb/88ou44447xODBg6Xl1dXVonfv3iImJkbs379ffPPNN8Lb21vMnTtXjl2yeXv27BFBQUGiT58+4umnn5baeZxb7uLFi6JLly5i8uTJYvfu3eL3338XW7duFSdPnpT6LFmyROh0OrFp0ybx66+/ivvuu08EBweLy5cvS31Gjhwp+vbtK37++Wfxv//9T4SGhorx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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.title(\"Stochastic gradient descent plot\")\n",
        "plt.xlabel(\"epoch\")\n",
        "plt.ylabel(\"error\")\n",
        "plt.plot(sgd_epoch_list,sgd_cost_list)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 490
        },
        "id": "KehRACevCGXE",
        "outputId": "c9636c51-ea3b-42a2-9bf7-1cc456d9d17d"
      },
      "execution_count": 741,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x78fe0dc41c90>]"
            ]
          },
          "metadata": {},
          "execution_count": 741
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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Fi9ejIiICAQFBem9V1paitLSUiiV+ptlZGQEnU5X7TrVajWsra31HlRZZ/e/90tsigYyzp8nIiKqF1kPgc2YMQMbNmzAF198gQsXLuCVV15Bfn4+wsLCAAATJkzAvHnzpP7Lly/HggUL8Nlnn8HX1xepqalITU1FXl4eAMDa2hoDBw7E7NmzERUVhatXr2Ljxo348ssv8fjjj8uyja2JpdoY8UuGAyi/OOKhSxkyV0RERFQ3st4KY+zYsbh16xYWLlyI1NRUBAYGIiIiQpoYnZSUpDeas27dOpSUlGD06NF661m0aBHeeustAMCWLVswb948jB8/HllZWfDx8cHSpUvx8ssvN9l2tWYqYyUsVEbIL9EifNdF9H/dSe6SiIiIak3W6wA1V7wO0N0t+DEWm44kwtPODIfmPCh3OURERABayHWAqOWaMrgdAOCmpgil2urnVhERETVXDEBUa85WaqiMldDqBG5m85pARETU8jAAUa0plQp425sDABKz8mWuhoiIqPYYgKhOfCoCUGaBzJUQERHVHgMQ1Ym3Q3kASspiACIiopaHAYjqpOIQWBJHgIiIqAViAKI68XGomAPEAERERC0PAxDVibe9BQAgKTOft8QgIqIWhwGI6sTTzgwKBZBfokVWfonc5RAREdUKAxDViamJEVytTQHwMBgREbU8DEBUZ5wITURELRUDENWZD0+FJyKiFooBiOrMmxdDJCKiFooBiOrM2+GvM8F4OwwiImphGICozng7DCIiaqkYgKjOKuYApecWo7BEK3M1RERENccARHVmY2YCK1NjAMBbO87hodW/4adTyTJXRUREdG8MQFRnCoVCmgi99c/ruJiai2+OJclcFRER0b0xAFG9dHCx0nt95EoWdDreGoOIiJo3BiCqlzce6oi3R3XB/pkDpbZ398bJWBEREdG9GctdALVsbjZmmNjHV6/tXEqOPMUQERHVEEeAqMF8HtYTABAVdwt5xWUyV0NERFQ9BiBqMHfOB9p8JFHGSoiIiO6u1gGotLQUQ4YMQUJCQmPUQy2Yu42p9DwhPU/GSoiIiO6u1gHIxMQEZ86caYxaqIVTKBRYPTYQAPD9iRsoKdPJWxAREVE16nQI7F//+hc+/fTThq6FWgGvv64LBAC7zt6UsRIiIqLq1ekssLKyMnz22WfYt28fevToAQsLC733V61a1SDF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z8e7776LP//8E8bGxpg0aVKTbD8RNSzeDJWImr3i4mLY29tj37596N27t9T+/PPPo6CgAC+++CIGDx6MLVu2YOzYsQCArKwseHp6YuPGjRgzZgzGjx+PW7duYc+ePdLyb7zxBnbu3Ilz584hPj4eHTt2xN69exESElKphqioKAwePBj79u3DkCFDAAC7du3CyJEjUVhYCFNT00beC0TUkDgCRETN3qVLl1BQUIChQ4fC0tJSenz55Ze4fPmy1O/OcGRvb4+OHTviwoULAIALFy6gb9++euvt27cvEhISoNVqcerUKRgZGWHgwIF3raVbt27Sczc3NwBAenp6vbeRiJqWsdwFEBHdS15eHgBg586d8PDw0HtPrVbrhaC6MjMzq1E/ExMT6blCoQBQPj+JiFoWjgARUbPXuXNnqNVqJCUloV27dnoPLy8vqd+RI0ek57dv30Z8fDw6deoEAOjUqRMOHz6st97Dhw+jQ4cOMDIyQteuXaHT6fTmFBFR68URICJq9qysrDBr1ixMnz4dOp0O/fr1g0ajweHDh2FtbQ0fHx8AwH/+8x84ODjAxcUF8+fPh6OjIx577DEAwMyZM9GzZ08sXrwYY8eORXR0NNasWYMPP/wQAODr64uJEydi0qRJeP/99xEQEIDExESkp6djzJgxcm06ETUSBiAiahEWL14MJycnhIeH48qVK7C1tUX37t3x5ptvSoegli1bhtdffx0JCQkIDAzEzz//DJVKBQDo3r07vv32WyxcuBCLFy+Gm5sb/vOf/+C5556TPmPdunV48803MXnyZGRmZsLb2xtvvvmmHJtLRI2MZ4ERUYtXcYbW7du3YWtrK3c5RNQCcA4QERERGRwGICIiIjI4PARGREREBocjQERERGRwGICIiIjI4DAAERERkcFhACIiIiKDwwBEREREBocBiIiIiAwOAxAREREZHAYgIiIiMjgMQERERGRw/h+t7nVnKsgVVAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Stochastic Gradient Descent has lesser accuracy then Batch Gradient Descent. This is because Batch Gradient Descent updates the weights by considering the entire batch every iteration. On the other hand, Stochastic Grasient Descent picks a training sample at random in each iteration in order to update the weights. As a result, Batch gradient descent systematically\n",
        "reduces error while stochastic gradient descent takes a more random approach.\n",
        "\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "CPVQwLarBcJY"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **_4. Comparison of Lasso and Ridge Regression using Polynomial Regression_**\n",
        "\n",
        "## **_Lasso Regression_**"
      ],
      "metadata": {
        "id": "2p-mPVFCSe9V"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 744,
      "metadata": {
        "id": "BBGotr9J9O6b"
      },
      "outputs": [],
      "source": [
        "class LassoReg():\n",
        "\n",
        "    # initializing the hyperparameters\n",
        "    def __init__(self, lr, iters, Lambda ,degree):\n",
        "        self.lr = lr\n",
        "        self.iters = iters\n",
        "        self.Lambda = Lambda\n",
        "        self.degree = degree\n",
        "\n",
        "    def generate_polynomial_features(X, degree):\n",
        "      #generating a polynomial in the form x+x^2+.....\n",
        "      X_poly = np.zeros((X.shape[0], degree))\n",
        "      for i in range(1, degree + 1):\n",
        "        X_poly[:, i - 1] = X ** i\n",
        "        return X_poly\n",
        "\n",
        "    def fit(self, X_poly, y):\n",
        "      #N---> no of data points = no of rows\n",
        "      #features---> no of input features = no of coloumn\n",
        "      self.N,self.features = X_poly.shape\n",
        "      #W = weights\n",
        "      #B = bias\n",
        "      self.W = np.zeros(self.features)\n",
        "      self.B = 0\n",
        "      self.X_poly = X_poly\n",
        "      self.y = y\n",
        "      # Implementing Gradient Descent algorithm for Optimization\n",
        "      for i in range(self.iters):\n",
        "        self.updateWeights()\n",
        "\n",
        "      #function  updating the value of weights and bias  for each value of x\n",
        "    def updateWeights(self):\n",
        "      y_pred = self.predict(self.X_poly)\n",
        "      dW = np.zeros(self.features)\n",
        "      # gradient loop for weights\n",
        "      for i in range(self.features):\n",
        "        if self.W[i] > 0:\n",
        "            dW[i] = (-(2 * (self.X_poly[:, i]).dot(self.y - y_pred)) + self.Lambda) / self.N\n",
        "        else:\n",
        "            dW[i] = (-(2 * (self.X_poly[:, i]).dot(self.y - y_pred)) - self.Lambda) / self.N\n",
        "\n",
        "            # gradient update for bias\n",
        "            dB = -2 * np.sum(self.y - y_pred) / self.N\n",
        "\n",
        "            self.W = self.W - self.lr * dW\n",
        "            self.B = self.B - self.lr * dB\n",
        "\n",
        "    # predicting the value of target variable\n",
        "    def predict(self, X_poly):\n",
        "      return X_poly.dot(self.W) + self.B\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 766,
      "metadata": {
        "id": "4ONMkF5ZC8wv"
      },
      "outputs": [],
      "source": [
        "dataset=(gdf-gdf.min())/(gdf.max()-gdf.min())\n",
        "X=dataset.drop(['Outcome', 'Unnamed: 0'], axis=1)\n",
        "y=dataset.Outcome\n",
        "dataset2=dataset.drop(['Unnamed: 0'], axis=1) #dataset without the 'Unnamed: 0' column\n",
        "\n",
        "X=dataset.iloc[:,:].values\n",
        "y=dataset.iloc[:,-1].values\n",
        "X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=9)\n",
        "model = LassoReg(lr=0.039, iters =1000,Lambda=1,  degree=3)\n",
        "#Training the model with training samples\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "#Making pradictions of the trained model using the tesing smaples\n",
        "y_pred = model.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "squared_error = (y_test - np.round(y_pred)) ** 2\n",
        "mse = np.mean(squared_error)\n",
        "\n",
        "print(f\"Mean Squared Error (MSE): {mse: .2f}\")"
      ],
      "metadata": {
        "id": "x6ZQ8xau2RWX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "53b52994-d30b-44ab-8644-db3f57f9de1e"
      },
      "execution_count": 767,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error (MSE):  0.36\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mse=mean_squared_error(y_test,y_pred)\n",
        "print(f\"Mean Squared Error (MSE): {mse: .2f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EwKEDSU28fMA",
        "outputId": "a6d07856-251e-43b2-c3a4-182138273958"
      },
      "execution_count": 768,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error (MSE):  0.23\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "As we can see, the difference in value of MSE calculated from scratch and that from the sklearn inbuilt function is negligible."
      ],
      "metadata": {
        "id": "2S4bnWXHjHw_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **_Ridge Regression_**"
      ],
      "metadata": {
        "id": "Dr-p_RtzSqUF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset=(gdf-gdf.min())/(gdf.max()-gdf.min())\n",
        "X=dataset.drop(['Outcome', 'Unnamed: 0'], axis=1)\n",
        "y=dataset.Outcome\n",
        "dataset2=dataset.drop(['Unnamed: 0'], axis=1) #dataset without the 'Unnamed: 0' column\n",
        "X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=9)"
      ],
      "metadata": {
        "id": "zaKWscUORaU-"
      },
      "execution_count": 769,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class RidgeReg:\n",
        "\n",
        "   # initializing the hyperparameters\n",
        "    def __init__(self, alpha, degree):\n",
        "        self.alpha = alpha\n",
        "        self.degree = degree\n",
        "        self.coef = None\n",
        "        self.intercept = None\n",
        "\n",
        "    #generating a polynomial in the form x+x^2+.....\n",
        "    def gen_poly_features(self, X):\n",
        "        X_poly = X.copy()\n",
        "        for i in range(2, self.degree + 1):\n",
        "            X_poly = np.concatenate((X_poly, X ** i), axis=1)\n",
        "        return X_poly\n",
        "\n",
        "    def fit(self, X_train, y_train):\n",
        "        X_train_poly = self.gen_poly_features(X_train)\n",
        "\n",
        "        X_train_modify = np.insert(X_train_poly, 0, 1, axis=1) # modifying the features (X) matrix to include a column of ones for the bias\n",
        "\n",
        "\n",
        "        I = np.identity(X_train_modify.shape[1])\n",
        "        I[0][0] = 0 #making the top leftmost element 0 to avoid penalising the y intercept\n",
        "\n",
        "        #calculating the ridge coeficient\n",
        "        b_rr = np.linalg.inv(np.dot(X_train_modify.T, X_train_modify) + self.alpha * I).dot(X_train_modify.T).dot(y_train)\n",
        "        self.intercept = b_rr[0]\n",
        "        self.coef = b_rr[1:]\n",
        "\n",
        "    def predict(self, X_test):\n",
        "        X_test_poly = self.gen_poly_features(X_test)\n",
        "        return np.dot(X_test_poly, self.coef) + self.intercept\n",
        "\n",
        "    #calculating mean squared error\n",
        "    def mse(self, y_true, y_pred):\n",
        "        return mean_squared_error(y_true, y_pred)\n"
      ],
      "metadata": {
        "id": "aWqMiCzVSuAD"
      },
      "execution_count": 770,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "degree_R=3\n",
        "ridge_reg = RidgeReg(alpha=0.039, degree=degree_R)\n",
        "\n",
        "#Training the model with training samples\n",
        "ridge_reg.fit(X_train, y_train)\n",
        "\n",
        "#Making pradictions of the trained model using the tesing smaples\n",
        "y_pred = ridge_reg.predict(X_test)\n",
        "\n",
        "#calculating the mean square error\n",
        "mse = ridge_reg.mse(y_test, y_pred)\n",
        "print(f\"Mean Squared Error (MSE): {mse: .2f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6xRkJbiM_DVZ",
        "outputId": "0aa89fff-e787-4569-b80f-f6f94c1f809e"
      },
      "execution_count": 771,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error (MSE):  0.19\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **_Insights drawn (plots, markdown explanations)_**"
      ],
      "metadata": {
        "id": "FlUMAWsiSug9"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "L1 and L2 regularisation are used to ensure no overfitting or underfitting occurs while performing polynomial regression.\n",
        "To determine the optimal values of learning rate and L1,L2 penalties we have run the model with different combinations and chosen the optimal set.\n",
        "From the models we have developed, the Mean Squared Error (MSE) for Lasso Regression () is 0.28 and that for Ridge Regression() it is 0.19.\n",
        "Possible reasons for this differnece could be that L1 handles outliers better as it can perform feature selection, unlike L2 in which the weights tend to 0 but are not exactly 0.\n"
      ],
      "metadata": {
        "id": "WZO4KIIMO5aR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "As observed from the correlation visualization,  Glucose is the feature that is the most correlated with the Outcome. Hence, we draw plots between the predicted value (Y) and Glucose (X)."
      ],
      "metadata": {
        "id": "rcY16J0rvJmu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_new=X_test['Glucose']"
      ],
      "metadata": {
        "id": "f5gJbpmdo1fl"
      },
      "execution_count": 772,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# LASSO Regression Plot\n",
        "plt.scatter(X_new, y_test, label=\"Test Data\", color=\"b\")\n",
        "plt.scatter(X_new, y_pred, label=\"Predictions\", color=\"r\")\n",
        "plt.scatter(X_new, y_pred.round(), label=\"Predicted values rounded off\", color=\"g\")\n",
        "plt.title(\"Polynomial Regression with Lasso Regularization\")\n",
        "plt.xlabel(\"X_test[Glucose]\")\n",
        "plt.ylabel(\"y_predicted\")\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "d8RRJbLCQPRk",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "outputId": "fbfda161-12dd-400e-e22c-6f15fe2398ff"
      },
      "execution_count": 773,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mymodel = np.poly1d(np.polyfit(X_new,y_pred,3))\n",
        "myline= np.linspace(1,10,100)\n",
        "plt.plot(myline,mymodel(myline))\n",
        "plt.scatter(X_new,y_pred,color='r')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 430
        },
        "id": "YK6SlYoUqXle",
        "outputId": "a24a261b-c41a-4026-f2f7-aa04910381dd"
      },
      "execution_count": 774,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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RESE4uLiNHjwYK1bt87qWA3GX//6VxmGoVGjRlkdJWDl5ubqlltuUZMmTRQaGqquXbtq8eLFVscKSG63W4899piSk5MVGhqqNm3a6M9//vMp9/VB7fz444+68sor1bx5cxmGoc8//7zG/aZp6vHHH1ezZs0UGhqq/v37a8OGDfWSjTJzHNOmTdPo0aM1fvx4LV26VN26dVNGRoZ27dpldbSAM2fOHI0cOVI//fSTMjMzVV5ergEDBqi4uNjqaAFv0aJFev3113XuuedaHSVg7d+/XxdccIGCg4M1Y8YM/fLLL5o4caKio6OtjhaQ/va3v+m1117TpEmTtHbtWv3tb3/TM888o5dfftnqaAGhuLhY3bp10yuvvHLc+5955hm99NJLmjx5sn7++WeFh4crIyNDJSUl3g9n4hi9evUyR44cWXXd7XabzZs3NydMmGBhqoZh165dpiRzzpw5VkcJaEVFRWa7du3MzMxM81e/+pV5//33Wx0pID388MNm3759rY7RYAwcONC8/fbba9x2zTXXmDfffLNFiQKXJPOzzz6ruu7xeMyEhATz2WefrbrtwIEDptPpND/88EOv52Fk5n+UlZVpyZIl6t+/f9VtNptN/fv3V1ZWloXJGobCwkJJUkxMjMVJAtvIkSM1cODAGn/OUff+85//KC0tTddff73i4uLUo0cP/eMf/7A6VsDq06ePZs2apfXr10uSVqxYoXnz5unyyy+3OFng27Jli/Lz82v8mxIVFaXevXvXy2dng9ho8nTs2bNHbrdb8fHxNW6Pj49Xdna2RakaBo/Ho1GjRumCCy5Qly5drI4TsP71r39p6dKlWrRokdVRAt7mzZv12muvafTo0frjH/+oRYsW6b777pPD4dCwYcOsjhdwxo4dK5fLpQ4dOshut8vtduvpp5/WzTffbHW0gJefny9Jx/3sPHqfN1Fm4DNGjhyp1atXa968eVZHCVg5OTm6//77lZmZqZCQEKvjBDyPx6O0tDT95S9/kST16NFDq1ev1uTJkykzXvDRRx/p/fff1wcffKDOnTtr+fLlGjVqlJo3b877HeD4mul/NG3aVHa7XQUFBTVuLygoUEJCgkWpAt8999yj6dOn6/vvv1fLli2tjhOwlixZol27dqlnz54KCgpSUFCQ5syZo5deeklBQUFyu91WRwwozZo1U6dOnWrc1rFjR23fvt2iRIFtzJgxGjt2rIYMGaKuXbtq6NCheuCBBzRhwgSrowW8o5+PVn12Umb+h8PhUGpqqmbNmlV1m8fj0axZs5Senm5hssBkmqbuueceffbZZ5o9e7aSk5OtjhTQLr30Uq1atUrLly+vuqSlpenmm2/W8uXLZbfbrY4YUC644IJjlhpYv369WrVqZVGiwHbo0CHZbDU/1ux2uzwej0WJGo7k5GQlJCTU+Ox0uVz6+eef6+Wzk6+ZjmP06NEaNmyY0tLS1KtXL73wwgsqLi7WbbfdZnW0gDNy5Eh98MEH+uKLLxQREVH13WpUVJRCQ0MtThd4IiIijpmPFB4eriZNmjBPyQseeOAB9enTR3/5y1/029/+VgsXLtSUKVM0ZcoUq6MFpCuvvFJPP/20kpKS1LlzZy1btkzPP/+8br/9dqujBYSDBw9q48aNVde3bNmi5cuXKyYmRklJSRo1apSeeuoptWvXTsnJyXrsscfUvHlzDR482PvhvH6+lJ96+eWXzaSkJNPhcJi9evUyf/rpJ6sjBSRJx728/fbbVkdrMDg127u+/PJLs0uXLqbT6TQ7dOhgTpkyxepIAcvlcpn333+/mZSUZIaEhJgpKSnmI488YpaWllodLSB8//33x/33etiwYaZpVp6e/dhjj5nx8fGm0+k0L730UnPdunX1ks0wTZZGBAAA/os5MwAAwK9RZgAAgF+jzAAAAL9GmQEAAH6NMgMAAPwaZQYAAPg1ygwAAPBrlBkAAODXKDMAAMCvUWYAAIBfo8wAAAC/RpkBAAB+7f8BUytq/3WM3BAAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Ridge Regression Plot\n",
        "plt.scatter(X_new, y_test, label=\"Test Data\", color=\"b\")\n",
        "plt.scatter(X_new, y_pred, label=\"Predictions\", color=\"r\")\n",
        "plt.scatter(X_new, y_pred.round(), label=\"Predicted values rounded off\", color=\"g\")\n",
        "plt.title(\"Polynomial Regression with Ridge Regression\")\n",
        "plt.xlabel(\"X_test[Glucose]\")\n",
        "plt.ylabel(\"y_predicted\")\n",
        "plt.legend()\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "6xc3AgHkQUJM",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "outputId": "23d7c0f7-4051-4761-ab62-9601b717155d"
      },
      "execution_count": 775,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mymodel = np.poly1d(np.polyfit(X_new,y_pred,degree_R))\n",
        "myline= np.linspace(1,30,100)\n",
        "plt.plot(myline,mymodel(myline))\n",
        "plt.scatter(X_new,y_pred,color='r')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 430
        },
        "id": "_LmZ1nujsBz3",
        "outputId": "15d70565-e6c6-42af-ac07-a435ff4e15c6"
      },
      "execution_count": 776,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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X30lXFxcxPvvv2/ezm+//SacnJzEG2+8IY4cOSLmzp0rlEqlOHjwoHnMggULhKenp/j+++/FgQMHxP333y/Cw8NFVVWVecywYcPELbfcInbu3Cl+/fVX0bFjRzF69Ogmvx+efUZERGR/mvP9bbVSdODAATF48GDh7e0t1Gq1CAsLExMnThRnzpyxGLd//34xYMAAoVarRbt27cSCBQuu2taqVatEp06dhEqlEt26dRNr1661WG8ymcTs2bNFQECAUKvVYsiQISInJ8dizIULF8To0aOFm5ub8PDwEOPHjxdlZWVNfj8sRURERPanOd/fMiHsYNazDTAYDNBqtdDr9fDw8JA6DhERETVBc76/mz3R2lFd7o68XhEREZH9uPy93ZRjQCxFTVRWVgYAvF4RERGRHSorK4NWq73mGP75rIlMJhMKCgrg7u4OmUx209szGAwIDg5GXl4e/xzXBNxfTcd91TzcX03HfdU83F/NY639JYRAWVkZgoKCIJdf+6R7HilqIrlcjvbt27f4dj08PPhflmbg/mo67qvm4f5qOu6r5uH+ah5r7K/rHSG6rNXufUZERERky1iKiIiIiMBSJBm1Wo25c+fyViJNxP3VdNxXzcP91XTcV83D/dU8trC/ONGaiIiICDxSRERERASApYiIiIgIAEsREREREQCWIiIiIiIALEWSeOeddxAWFgaNRoPY2Fjs2rVL6kg2ad68eZDJZBZLVFSU1LFsxrZt23DvvfciKCgIMpkM3333ncV6IQTmzJmDwMBAODs7Iz4+HkePHpUmrA243v4aN27cVZ+3YcOGSRNWYvPnz0efPn3g7u4Of39/JCYmIicnx2JMdXU1Jk2aBB8fH7i5uWHEiBEoKiqSKLF0mrKvBg0adNVna+LEiRIlltZ7772Hnj17mi/QGBcXh59++sm8XurPFUtRK1u5ciWSk5Mxd+5c7N27F7169UJCQgKKi4uljmaTunXrhrNnz5qXX3/9VepINqOiogK9evXCO++80+j6hQsXYunSpUhJScHOnTvh6uqKhIQEVFdXt3JS23C9/QUAw4YNs/i8ffnll62Y0HZs3boVkyZNwo4dO5Camoq6ujoMHToUFRUV5jEvvPACfvzxR6xevRpbt25FQUEBHnzwQQlTS6Mp+woAJkyYYPHZWrhwoUSJpdW+fXssWLAAGRkZ2LNnD+68807cf//9yMrKAmADnytBrapv375i0qRJ5sdGo1EEBQWJ+fPnS5jKNs2dO1f06tVL6hh2AYD49ttvzY9NJpPQ6XRi0aJF5udKS0uFWq0WX375pQQJbcuf95cQQowdO1bcf//9kuSxdcXFxQKA2Lp1qxCi4bOkVCrF6tWrzWOOHDkiAIj09HSpYtqEP+8rIYS44447xPPPPy9dKBvn5eUlPvroI5v4XPFIUSuqra1FRkYG4uPjzc/J5XLEx8cjPT1dwmS26+jRowgKCkJERATGjBmD3NxcqSPZhZMnT6KwsNDis6bVahEbG8vP2jVs2bIF/v7+6Ny5M5599llcuHBB6kg2Qa/XAwC8vb0BABkZGairq7P4fEVFRSEkJMThP19/3leXff755/D19UX37t0xa9YsVFZWShHPphiNRnz11VeoqKhAXFycTXyueEPYVnT+/HkYjUYEBARYPB8QEIDs7GyJUtmu2NhYrFixAp07d8bZs2fxyiuvYODAgTh06BDc3d2ljmfTCgsLAaDRz9rldWRp2LBhePDBBxEeHo7jx4/j5Zdfxl133YX09HQoFAqp40nGZDJh6tSpuO2229C9e3cADZ8vlUoFT09Pi7GO/vlqbF8BwKOPPorQ0FAEBQXhwIEDmDFjBnJycvDNN99ImFY6Bw8eRFxcHKqrq+Hm5oZvv/0WXbt2RWZmpuSfK5Yisll33XWX+feePXsiNjYWoaGhWLVqFZKSkiRMRm3RqFGjzL/36NEDPXv2RIcOHbBlyxYMGTJEwmTSmjRpEg4dOsT5fE3wV/vq6aefNv/eo0cPBAYGYsiQITh+/Dg6dOjQ2jEl17lzZ2RmZkKv1+Prr7/G2LFjsXXrVqljAeBE61bl6+sLhUJx1Uz6oqIi6HQ6iVLZD09PT3Tq1AnHjh2TOorNu/x54mftxkVERMDX19ehP2+TJ0/GmjVrsHnzZrRv3978vE6nQ21tLUpLSy3GO/Ln66/2VWNiY2MBwGE/WyqVCpGRkYiJicH8+fPRq1cvvPXWWzbxuWIpakUqlQoxMTFIS0szP2cymZCWloa4uDgJk9mH8vJyHD9+HIGBgVJHsXnh4eHQ6XQWnzWDwYCdO3fys9ZEZ86cwYULFxzy8yaEwOTJk/Htt99i06ZNCA8Pt1gfExMDpVJp8fnKyclBbm6uw32+rrevGpOZmQkADvnZaozJZEJNTY1NfK7457NWlpycjLFjx6J3797o27cvlixZgoqKCowfP17qaDbnxRdfxL333ovQ0FAUFBRg7ty5UCgUGD16tNTRbEJ5ebnF/9M8efIkMjMz4e3tjZCQEEydOhWvvvoqOnbsiPDwcMyePRtBQUFITEyULrSErrW/vL298corr2DEiBHQ6XQ4fvw4pk+fjsjISCQkJEiYWhqTJk3CF198ge+//x7u7u7m+RxarRbOzs7QarVISkpCcnIyvL294eHhgSlTpiAuLg79+vWTOH3rut6+On78OL744gvcfffd8PHxwYEDB/DCCy/g9ttvR8+ePSVO3/pmzZqFu+66CyEhISgrK8MXX3yBLVu2YMOGDbbxuWqVc9zIwrJly0RISIhQqVSib9++YseOHVJHskkjR44UgYGBQqVSiXbt2omRI0eKY8eOSR3LZmzevFkAuGoZO3asEKLhtPzZs2eLgIAAoVarxZAhQ0ROTo60oSV0rf1VWVkphg4dKvz8/IRSqRShoaFiwoQJorCwUOrYkmhsPwEQy5cvN4+pqqoSzz33nPDy8hIuLi7igQceEGfPnpUutESut69yc3PF7bffLry9vYVarRaRkZHipZdeEnq9XtrgEnnyySdFaGioUKlUws/PTwwZMkT8/PPP5vVSf65kQgjROvWLiIiIyHZxThERERERWIqIiIiIALAUEREREQFgKSIiIiICwFJEREREBICliIiIiAgASxERERERAJYiIiIiIgAsRUREREQAWIqIiIiIALAUEREREQFgKSIiIiICAPx/PX8crabg+jUAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **_5. Comparison of Logistic Regression and Least Squares Classification_**\n",
        "\n",
        "## **_Logistic Regression_**"
      ],
      "metadata": {
        "id": "jzSfpXtdS_q0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset=(gdf-gdf.min())/(gdf.max()-gdf.min())\n",
        "X=dataset.drop(['Outcome', 'Unnamed: 0'], axis=1)\n",
        "y=dataset.Outcome\n",
        "dataset2=dataset.drop(['Unnamed: 0'], axis=1) #dataset without the 'Unnamed: 0' column\n",
        "X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=50)"
      ],
      "metadata": {
        "id": "6A8f5lnlRb5o"
      },
      "execution_count": 777,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#using sigmoid function as the activation function\n",
        "def sigmoid(X):\n",
        "    return 1/(1+np.exp(-X))\n",
        "\n",
        "#creating logistic regression model from scratch\n",
        "class LogReg():\n",
        "    #constructor for initialising paramemters\n",
        "    def __init__(self,lr=0.01, iters=8000):\n",
        "        self.lr=lr\n",
        "        self.iters=iters\n",
        "        self.Weights=None\n",
        "        self.Bias=None\n",
        "\n",
        "    #method to train and fit the training data\n",
        "    def fit(self,X,y):\n",
        "        N, features = X.shape\n",
        "        self.Weights=np.zeros(features)\n",
        "        self.Bias=0\n",
        "\n",
        "        for i in range(self.iters):\n",
        "            linear_pred = np.dot(X, self.Weights)+self.Bias\n",
        "            predictions=sigmoid(linear_pred)\n",
        "\n",
        "            dW = (1/N)*np.dot(X.T,predictions-y)\n",
        "            dB = (1/N)*np.sum(predictions-y)\n",
        "\n",
        "            self.Weights=self.Weights-self.lr*dW\n",
        "            self.Bias=self.Bias=self.lr*dB\n",
        "\n",
        "    #method to predict outcome on testing data after model is trained\n",
        "    def predict(self,X):\n",
        "        linear_pred = np.dot(X, self.Weights)+self.Bias\n",
        "        y_pred=sigmoid(linear_pred)\n",
        "        final_pred=[0 if y<0.5 else 1 for y in y_pred]\n",
        "        return final_pred"
      ],
      "metadata": {
        "id": "JLCup6QlTG0G"
      },
      "execution_count": 787,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "logReg=LogReg()\n",
        "#Training the model with training samples\n",
        "logReg.fit(X_train,y_train)\n",
        "#Making pradictions of the trained model using the tesing smaples\n",
        "y_pred=logReg.predict(X_test)\n"
      ],
      "metadata": {
        "id": "8oZidUR_sDrr"
      },
      "execution_count": 788,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#evaluating efficiency of model\n",
        "def accuracy(y_pred,y_test):\n",
        "    return np.sum(y_pred==y_test)/len(y_test)\n",
        "acc=accuracy(y_pred,y_test)\n",
        "print(acc)"
      ],
      "metadata": {
        "id": "-prjlvmSy5-m",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "62df6a2b-e5db-4ffa-97fd-617625437971"
      },
      "execution_count": 790,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.59\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "acc=accuracy_score(y_test,y_pred)\n",
        "print(f\"F1 score: {acc: .2f}\")\n",
        "print(f\"F1 score: {f1_score(y_test,y_pred): .2f}\")"
      ],
      "metadata": {
        "id": "OjZc5pC_8XVa",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "29a76bff-60d4-4215-f99f-8b2feaaa4d1d"
      },
      "execution_count": 791,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "F1 score:  0.59\n",
            "F1 score:  0.28\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "As we can see, the difference in accuracy calculated from scratch and that from the sklearn inbuilt function is negligible.\n"
      ],
      "metadata": {
        "id": "auVsAKRUzGTi"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **_Least Squares Classification_**"
      ],
      "metadata": {
        "id": "kdX2kQ-nTHXc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset=(gdf-gdf.min())/(gdf.max()-gdf.min())\n",
        "X=dataset.drop(['Outcome', 'Unnamed: 0'], axis=1)\n",
        "y=dataset.Outcome\n",
        "dataset2=dataset.drop(['Unnamed: 0'], axis=1) #dataset without the 'Unnamed: 0' column\n",
        "X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=9)"
      ],
      "metadata": {
        "id": "5GAoSjWyRepe"
      },
      "execution_count": 792,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class LeastSquareClassification():\n",
        "    def __init__(self):\n",
        "        self.t0 = 20\n",
        "        self.t1 = 1000\n",
        "\n",
        "    def predict(self, X):\n",
        "        if X.shape[-1] != self.w.shape[0]:\n",
        "          print(\"Shape of x and w not complatible\")\n",
        "        #assert X.shape[-1] == self.w.shape[0], f\"X shape{X.shape} and w shape {self.w.shape}, are not compatible\"\n",
        "        return X @ self.w\n",
        "\n",
        "    def loss(self, X, y, reg_rate):\n",
        "        e = y - self.predict_internal(X)\n",
        "        return (1/2) * (e.T@e) + (reg_rate/2)*(self.w).T@(self.w)\n",
        "\n",
        "\n",
        "    def predict_internal(self, X):\n",
        "        if X.shape[-1] != self.w.shape[0]:\n",
        "          print(\"Shape of x and w not complatible\")\n",
        "        #assert X.shape[-1] == self.w.shape[0], f\"X shape {X.shape} and w shape {self.w.shape}, are not compatible\"\n",
        "        return X@self.w\n",
        "\n",
        "    def rmse(self, X, y):\n",
        "        return np.sqrt(2/X.shape[0] * self.loss(X, y, 0))\n",
        "\n",
        "    def fit(self, X, y, reg_rate):\n",
        "        eye = np.eye(X.shape[1])\n",
        "        self.w = np.linalg.solve(reg_rate*eye + X.T@X, X.T @ y)\n",
        "        return self.w\n",
        "\n",
        "    def calculate_gradient(self, X, y, reg_rate):\n",
        "        grad = X.T @  (self.predict_internal(X)-y) + reg_rate * self.w\n",
        "        return grad\n",
        "\n",
        "    def update_weight(self, grad, lr) -> np.ndarray:\n",
        "        return(self.w - lr*grad)\n",
        "\n",
        "    def learning_schedule(self, t):\n",
        "        return self.t0/(t+self.t1)\n",
        "\n",
        "    def gd(self, X, y, epochs, lr, reg_rate):\n",
        "        self.w = np.zeros(X.shape[1])\n",
        "        self.w_all = []\n",
        "        self.err_all = []\n",
        "        for i in np.arange(0, epochs):\n",
        "            djdw = self.calculate_gradient(X, y, reg_rate)\n",
        "            lr = self.learning_schedule(epochs*X.shape[0]+i)\n",
        "            self.w_all.append(self.w)\n",
        "            self.err_all.append(self.loss(X, y, reg_rate))\n",
        "            self.w = self.update_weights(djdw, lr)\n",
        "        return self.w"
      ],
      "metadata": {
        "id": "_7qJtwGnTMdb"
      },
      "execution_count": 793,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lsc = LeastSquareClassification()\n",
        "lsc.fit(X_train,y_train,0.39)\n",
        "y_pred=lsc.predict(X_test)\n",
        "y_pred=np.round(y_pred)\n",
        "def accuracy(y_pred,y_test):\n",
        "    return np.sum(y_pred==y_test)/len(y_test)\n",
        "acc=accuracy(y_pred,y_test)\n",
        "print(f\"Accuracy: {acc: .2f}\")\n",
        "acc=accuracy_score(y_test,y_pred)\n",
        "print(f\"Accuracy: {acc: .2f}\")\n",
        "print(f\"F1 score: {f1_score(y_test,y_pred): .2f}\")\n"
      ],
      "metadata": {
        "id": "TlPggU6OsLgx",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "de3ba1cf-8ede-4b49-ff17-78e83a757827"
      },
      "execution_count": 794,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy:  0.70\n",
            "Accuracy:  0.70\n",
            "F1 score:  0.48\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **_Insights drawn (plots, markdown explanations)_**"
      ],
      "metadata": {
        "id": "WSoa7KO1TM6-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.title(\"Sigmoid function used in Logistic Regression\")\n",
        "y=sigmoid(X_test)\n",
        "plt.plot(X_test,y)\n",
        "plt.xlabel(\"X\")\n",
        "plt.ylabel(\"Sigmoid f(x)\")"
      ],
      "metadata": {
        "id": "9KYE7sstTW4D",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 490
        },
        "outputId": "037e1e32-cdad-4c42-fd96-e2e29e97bb91"
      },
      "execution_count": 795,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0, 0.5, 'Sigmoid f(x)')"
            ]
          },
          "metadata": {},
          "execution_count": 795
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Data visualisation for Least Square Classification\n",
        "confusion_matrix = metrics.confusion_matrix(y_test, y_pred)\n",
        "cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])\n",
        "cm_display.plot()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 449
        },
        "id": "o4HgFv628ZsS",
        "outputId": "f2852aee-41da-4155-be25-4bc438bc9a5e"
      },
      "execution_count": 796,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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jBs3Tj179nSuOxwOZWRkqFWrVgoPD1eXLl306quv1u6HA/4DCQNqVVBQkDIzM7V//34tW7ZM7777riZOnHjF44cMGaLmzZtr165dysnJ0eTJkxUSEiJJOnLkiPr06aO0tDR9/PHHWr16tbKzszV69Oja+jhAtUyePFlz5szRgQMH1Llz52qdk5GRoeXLl2vRokXav3+/0tPTdd999ykrK8vH0QKXR10MPrNx40aXcmzfvn21du1a5/q1116r2bNn66GHHtKCBQsue428vDw9/vjjateunSSpTZs2zn0ZGRkaMmSIxo0b59yXmZmpHj16aOHChQH7AhjUPzNnztQdd9xR7ePLysr01FNP6Z133lFKSookqXXr1srOztaLL76oHj16+CpU4IpIGOAzvXr10sKFC53rEREReuedd5SRkaGDBw+qqKhIlZWVKi0t1fnz59WwYcNLrjF+/HiNHDlSr7zyinr37q1f/epXuu666yRdaFd8/PHHWrFihfN4wzDkcDh07NgxtW/f3vcfEqiGG2+80aPjDx8+rPPnz1+SZJSXl6tr165mhgZUGwkDfCYiIkKJiYnO9c8//1y//OUv9fDDD+vJJ59UTEyMsrOzNWLECJWXl182YZg+fbp+85vfaNOmTXrzzTc1bdo0rVq1SgMGDFBxcbF+97vfaezYsZec16JFC59+NsATERERLutBQUH6/qz8FRUVzp8vjuvZtGmTmjVr5nIc759AXSFhQK3JycmRw+HQs88+q6CgC8Nn1qxZ4/a8pKQkJSUlKT09XYMHD9aSJUs0YMAAdevWTZ9++qlLUgLUB02aNNG+fftctu3Zs8c5PqdDhw6yWq3Ky8uj/QC/waBH1JrExERVVFTohRde0NGjR/XKK69o0aJFVzz+22+/1ejRo7V161Z98cUX2rZtm3bt2uVsNUyaNEnbt2/X6NGjtWfPHh06dEivv/46gx7h92677Tbt3r1by5cv16FDhzRt2jSXBCIqKkoTJkxQenq6li1bpiNHjujDDz/UCy+8oGXLltVh5LiakTCg1nTp0kVz587VH//4R3Xs2FErVqxQRkbGFY8PDg7W6dOnNXToUCUlJWnQoEHq27evZsyYIUnq3LmzsrKy9Nlnn6l79+7q2rWrpk6dqvj4+Nr6SECNpKamasqUKZo4caJ+/OMf69y5cxo6dKjLMbNmzdKUKVOUkZGh9u3bq0+fPtq0aZNatWpVR1HjasfrrQEAgFtUGAAAgFskDAAAwC0SBgAA4BYJAwAAcIuEAQAAuEXCAAAA3CJhAAAAbpEwAAAAt0gYgHpi+PDh6t+/v3O9Z8+ezld716atW7fKYrHo7NmzVzzGYrFo/fr11b7m9OnTdcMNN3gV1+effy6LxaI9e/Z4dR0Al0fCAHhh+PDhslgsslgsCg0NVWJiombOnKnKykqf3/tvf/ubZs2aVa1jq/MlDwA/hLdVAl7q06ePlixZorKyMr3xxhsaNWqUQkJC9MQTT1xybHl5uUJDQ025b0xMjCnXAYDqoMIAeMlqtSouLk4tW7bUww8/rN69e+vvf/+7pH+3EZ588knFx8erbdu2kqTjx49r0KBBio6OVkxMjPr166fPP//cec2qqiqNHz9e0dHRaty4sSZOnKjvv/bl+y2JsrIyTZo0SQkJCbJarUpMTNTixYv1+eefq1evXpKkRo0ayWKxaPjw4ZIkh8OhjIwMtWrVSuHh4erSpYteffVVl/u88cYbSkpKUnh4uHr16uUSZ3VNmjRJSUlJatiwoVq3bq0pU6aooqLikuNefPFFJSQkqGHDhho0aJAKCwtd9r/88stq3769wsLC1K5dOy1YsMDjWADUDAkDYLLw8HCVl5c717ds2aLc3Fxt3rxZGzduVEVFhVJTUxUVFaX3339f27ZtU2RkpPr06eM879lnn9XSpUv15z//WdnZ2Tpz5ozWrVv3g/cdOnSo/vrXvyozM1MHDhzQiy++qMjISCUkJOi1116TJOXm5io/P1/PP/+8JCkjI0PLly/XokWLtH//fqWnp+u+++5TVlaWpAuJzcCBA3XXXXdpz549GjlypCZPnuzx/yZRUVFaunSpPv30Uz3//PN66aWXNG/ePJdjDh8+rDVr1mjDhg1666239NFHH+mRRx5x7l+xYoWmTp2qJ598UgcOHNBTTz2lKVOm8LpnoLYYAGps2LBhRr9+/QzDMAyHw2Fs3rzZsFqtxoQJE5z7Y2NjjbKyMuc5r7zyitG2bVvD4XA4t5WVlRnh4eHG22+/bRiGYVxzzTXG008/7dxfUVFhNG/e3HkvwzCMHj16GI8++qhhGIaRm5trSDI2b9582Tj/+c9/GpKMb775xrmttLTUaNiwobF9+3aXY0eMGGEMHjzYMAzDeOKJJ4wOHTq47J80adIl1/o+Sca6deuuuP+ZZ54xkpOTnevTpk0zgoODjRMnTji3vfnmm0ZQUJCRn59vGIZhXHfddcbKlStdrjNr1iwjJSXFMAzDOHbsmCHJ+Oijj654XwA1xxgGwEsbN25UZGSkKioq5HA49Jvf/EbTp0937u/UqZPLuIW9e/fq8OHDioqKcrlOaWmpjhw5osLCQuXn5+umm25y7mvQoIFuvPHGS9oSF+3Zs0fBwcHq0aNHteM+fPiwzp8/rzvuuMNle3l5ubp27SpJOnDggEsckpSSklLte1y0evVqZWZm6siRIyouLlZlZaVsNpvLMS1atFCzZs1c7uNwOJSbm6uoqCgdOXJEI0aM0IMPPug8prKyUna73eN4AHiOhAHwUq9evbRw4UKFhoYqPj5eDRq4/mcVERHhsl5cXKzk5GStWLHikms1adKkRjGEh4d7fE5xcbEkadOmTS5f1NKFcRlm2bFjh4YMGaIZM2YoNTVVdrtdq1at0rPPPutxrC+99NIlCUxwcLBpsQK4MhIGwEsRERFKTEys9vHdunXT6tWr1bRp00v+yr7ommuu0QcffKBbb71V0oW/pHNyctStW7fLHt+pUyc5HA5lZWWpd+/el+y/WOGoqqpybuvQoYOsVqvy8vKuWJlo3769cwDnRTt37nT/If/D9u3b1bJlS/33f/+3c9sXX3xxyXF5eXk6efKk4uPjnfcJCgpS27ZtFRsbq/j4eB09elRDhgzx6P4AzMGgR6CWDRkyRD/60Y/Ur18/vf/++zp27Ji2bt2qsWPH6sSJE5KkRx99VHPmzNH69et18OBBPfLIIz84h8K1116rYcOG6YEHHtD69eud11yzZo0kqWXLlrJYLNq4caO+/vprFRcXKyoqShMmTFB6erqWLVumI0eO6MMPP9QLL7zgHEj40EMP6dChQ3r88ceVm5urlStXaunSpR593jZt2igvL0+rVq3SkSNHlJmZedkBnGFhYRo2bJj27t2r999/X2PHjtWgQYMUFxcnSZoxY4YyMjKUmZmpzz77TJ988omWLFmiuXPnehQPgJohYQBqWcOGDfXee++pRYsWGjhwoNq3b68RI0aotLTUWXF47LHH9Nvf/lbDhg1TSkqKoqKiNGDAgB+87sKFC3XPPffokUceUbt27fTggw+qpKREktSsWTPNmDFDkydPVmxsrEaPHi1JmjVrlqZMmaKMjAy1b99effr00aZNm9SqVStJF8YVvPbaa1q/fr26dOmiRYsW6amnnvLo8959991KT0/X6NGjdcMNN2j79u2aMmXKJcclJiZq4MCB+vnPf64777xTnTt3dnlscuTIkXr55Ze1ZMkSderUST169NDSpUudsQLwLYtxpVFUAAAA36HCAAAA3CJhAAAAbpEwAAAAt0gYAACAWyQMAADALRIGAADgFgkDAABwi4QBAAC4RcIAAADcImEAAABukTAAAAC3/h9v/9W5JAiZZgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "As seen from the models the accuracy of both Logistic regression and Least square classification is around 70% with least square classification performing slightly better according to samples of dataset provided"
      ],
      "metadata": {
        "id": "FtI1-QrnSp_o"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **_5. References_**\n",
        "\n",
        "1.   Christopher Bishop: Pattern Recognition and Machine Learning, Springer International Edition (Chapter 3,4)\n",
        "2.  [Ridge and Lasso Regression Indepth Intuition](https://www.youtube.com/watch?v=9lRv01HDU0s)\n",
        "3. [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)\n",
        "4. [ Logistic Regression Indepth Intuition](https://www.youtube.com/watch?v=L_xBe7MbPwk)\n",
        "5. [Stochastic Gradient Descent vs Batch Gradient Descent vs Mini Batch Gradient Descent](https://youtu.be/IU5fuoYBTAM?feature=shared)\n",
        "6. [NumPy documentation](https://numpy.org/doc/)\n",
        "7. [Pandas documentation](https://pandas.pydata.org/docs/)\n",
        "8. [matplotlib documentation](https://matplotlib.org/stable/index.html)\n",
        "9.\n"
      ],
      "metadata": {
        "id": "81SwQ_y4TaSc"
      }
    }
  ]
}
