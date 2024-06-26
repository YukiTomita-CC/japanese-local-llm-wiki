# Japanese Local LLM Wiki

## Notice
このリポジトリはローカルLLMについての日本語wikiを作成するためのものです。  
This repository is to create a Japanese wiki about local LLM.

これはまだ構想段階のため頓挫する可能性が大いにありますが、~~2024年6月末~~に一先ずの公開を目標にしています。  
Since this project is still in the conceptual stage, there is a possibility of setbacks. However, the goal is to have an initial public release by ~~the end of June 2024~~.

もし記事に間違いや不足がありましたらIssueやPRでお知らせいただけますと幸いです。  
If you find any mistakes or omissions in the articles, please let me know via an Issue or PR.

このリポジトリは[Docusaurus](https://docusaurus.io/)を使用しています。  
This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

**June 27, 2024 Update**  
私はLLMを理解したつもりでいましたが、何も理解していませんでした。  
I thought I understood LLMs, but I realized I didn't understand anything.  

「transformer」という言葉が何を指しているのか？「学習」とは何か？  
What does the term "transformer" mean? What is "train"?

今私は線形代数と機械学習の基礎からやり直しています。  
I am starting over from the basics of linear algebra and machine learning.

新たな目標として、7月末までにwikiの公開を目指します。  
As a new goal, I aim to release the wiki by the end of July 2024.

### Installation

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

Using SSH:

```
$ USE_SSH=true yarn deploy
```

Not using SSH:

```
$ GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
