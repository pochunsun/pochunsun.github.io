---
layout: post
title: Coverting LaTeX to HTML
description: "Here you can enter the LaTeX code, and then you will get the corresponding HTML code."
tags: [LaTeX,HTML]
image:
  background: triangular.png
  
modified: 2021-02-02
---

<label for="input">Latex equation: </label>
<!-- <input type="text" name="input"> -->
<br><textarea rows="10" cols="50" name="input"></textarea>
<br><br>

<button onClick="toGithubRenderURL()">Encode to github render URL</button>
<br><br>

<label for="output">The URL is: </label>
<input type="url" name="output">
<br><br>

<label for="result">The result is: </label>
<p id="result"></p>
