# Multi-Energy Load Forecasting with Blockchain-Assisted Secure Vertical Federated Learning

## Abstract

Load forecasting in multi-energy systems (MES) is crucial for optimizing resource allocation and enhancing energy efficiency. By leveraging data from various sectors like cooling, heating, and electricity, forecast accuracy can improve; however, this raises privacy concerns. Federated Learning (FL) offers a solution, but requires a secure training environment for reliable model updates and data aggregation. This paper introduces a Blockchain-assisted secure Vertical Federated Learning framework (BC-sVFL) for MES load forecasting. It enhances privacy via an improved loss function to avoid reverting smashed data. Functional Encryption (FE) secures data transmission, while blockchain ensures automated, reliable training protocols and key exchanges, preventing collusion and providing a traceable environment. Experimental evaluations confirm the framework's effectiveness.

## Problem Statement

In multi-energy systems, load data from various sectors is critical for forecasting and improving system efficiency. However, the proprietary nature of this data and potential privacy concerns hinder sharing among different sectors, like cooling, heating, and electricity. Traditional methods relying on centralized data can compromise privacy, necessitating a secure approach to jointly utilize data across sectors without compromising ownership or confidentiality.

![Problems in current practices](images/ps.png)

## Schemes

The BC-sVFL framework introduces several key components and mechanisms:

- **Enhanced Privacy via Invertibility Reduction**: An improved loss function is incorporated to the Vertical Federated Learning framework, minimizing the risk of reverse-engineering local data from smashed data.
- **Functional Encryption (FE)**: Utilized to ensure secure data transmission between clients and cloud servers, safeguarding sensitive information throughout the communication process.
- **Blockchain Integration**: The blockchain plays a pivotal role in automating and securing the execution of training protocols and key exchanges. This prevents potential collusion between participating entities while maintaining an auditable and transparent training environment.

## Code Explanation

The code for this project is organized into several main sections:

- **Data Preprocessing**: This module handles the initial preparation of data from different sectors, ensuring compatibility for federated learning models.
- **Federated Learning Module**: Implements the core VFL model, integrating privacy-preserving mechanisms like the improved loss function and functional encryption.
- **Blockchain Layer**: Facilitates secure training protocol automation and key exchanges, enhancing model integrity and accountability.
- **Simulation and Evaluation**: Provides scripts and tools to test and validate the BC-sVFL framework under various scenarios, demonstrating its effectiveness and efficiency through simulated datasets and metrics.

To get started, clone the repository and follow the installation instructions provided in the documentation. Detailed comments and documentation are included within the code to guide further implementation and customization.
